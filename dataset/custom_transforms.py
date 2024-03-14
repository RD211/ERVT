import numpy as np
import torch
from tonic.slicers import (
    slice_events_by_time,
)
from tonic.functional import to_voxel_grid_numpy
from typing import Any, List, Tuple
import torch as th
from dataclasses import dataclass
import argparse
import matplotlib.pyplot as plt
import cv2

class SliceByTimeEventsTargets:
    """
    Modified from tonic.slicers.SliceByTimeEventsTargets in the Tonic Library

    Slices an event array along fixed time window and overlap size. The number of bins depends
    on the length of the recording. Targets are copied.

    >        <overlap>
    >|    window1     |
    >        |   window2     |

    Parameters:
        time_window (int): time for window length (same unit as event timestamps)
        overlap (int): overlap (same unit as event timestamps)
        include_incomplete (bool): include the last incomplete slice that has shorter time
    """

    def __init__(self,time_window, overlap=0.0, seq_length=30, seq_stride=15, include_incomplete=False) -> None:
        self.time_window= time_window
        self.overlap= overlap
        self.seq_length=seq_length
        self.seq_stride=seq_stride
        self.include_incomplete=include_incomplete

    def slice(self, data: np.ndarray, targets: int) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data, targets)
        return self.slice_with_metadata(data, targets, metadata)

    def get_slice_metadata(
        self, data: np.ndarray, targets: int
    ) -> List[Tuple[int, int]]:
        t = data["t"]
        stride = self.time_window - self.overlap
        assert stride > 0

        if self.include_incomplete:
            n_slices = int(np.ceil(((t[-1] - t[0]) - self.time_window) / stride) + 1)
        else:
            n_slices = int(np.floor(((t[-1] - t[0]) - self.time_window) / stride) + 1)
        n_slices = max(n_slices, 1)  # for strides larger than recording time

        window_start_times = np.arange(n_slices) * stride + t[0]
        window_end_times = window_start_times + self.time_window
        indices_start = np.searchsorted(t, window_start_times)[:n_slices]
        indices_end = np.searchsorted(t, window_end_times)[:n_slices]

        if not self.include_incomplete:
            # get the strided indices for loading labels
            label_indices_start = np.arange(0, targets.shape[0]-self.seq_length, self.seq_stride)
            label_indices_end = label_indices_start + self.seq_length
        else:
            label_indices_start = np.arange(0, targets.shape[0], self.seq_stride)
            label_indices_end = label_indices_start + self.seq_length
            # the last label indices end should be the last label
            label_indices_end[-1] = targets.shape[0]

        assert targets.shape[0] >= label_indices_end[-1]

        return list(zip(zip(indices_start, indices_end), zip(label_indices_start, label_indices_end)))

    @staticmethod
    def slice_with_metadata(
        data: np.ndarray, targets: int, metadata: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ):
        return_data = []
        return_target = []
        for tuple1, tuple2 in metadata:
            return_data.append(data[tuple1[0]:tuple1[1]])
            return_target.append(targets[tuple2[0]:tuple2[1]])

        return return_data, return_target


class SliceLongEventsToShort:
    def __init__(self, time_window, overlap, include_incomplete):
        """
        Initialize the transformation.

        Args:
        - time_window (int): The length of each sub-sequence.
        """
        self.time_window = time_window
        self.overlap = overlap
        self.include_incomplete = include_incomplete

    def __call__(self, events):
        return slice_events_by_time(events, self.time_window, self.overlap, self.include_incomplete)


class EventSlicesToVoxelGrid:
    def __init__(self, sensor_size, n_time_bins, per_channel_normalize):
        """
        Initialize the transformation.

        Args:
        - sensor_size (tuple): The size of the sensor.
        - n_time_bins (int): The number of time bins.
        """
        self.sensor_size = sensor_size
        self.n_time_bins = n_time_bins
        self.per_channel_normalize = per_channel_normalize

    def __call__(self, event_slices):
        """
        Apply the transformation to the given event slices.

        Args:
        - event_slices (Tensor): The input event slices.

        Returns:
        - Tensor: A batched tensor of voxel grids.
        """
        voxel_grids = []
        for event_slice in event_slices:
            voxel_grid = to_voxel_grid_numpy(event_slice, self.sensor_size, self.n_time_bins)
            voxel_grid = voxel_grid.squeeze(-3)
            if self.per_channel_normalize:
                # Calculate mean and standard deviation only at non-zero values
                non_zero_entries = (voxel_grid != 0)
                for c in range(voxel_grid.shape[0]):
                    mean_c = voxel_grid[c][non_zero_entries[c]].mean()
                    std_c = voxel_grid[c][non_zero_entries[c]].std()

                    voxel_grid[c][non_zero_entries[c]] = (voxel_grid[c][non_zero_entries[c]] - mean_c) / (std_c + 1e-10)
            voxel_grids.append(voxel_grid)
        return np.array(voxel_grids).astype(np.float32)

class EventSlicesToSpikeTensor:
    def __init__(self, sensor_size, n_time_bins, per_channel_normalize):
        """
        Initialize the transformation.

        Args:
        - sensor_size (tuple): The size of the sensor.
        - n_time_bins (int): The number of time bins.
        """

        self.sensor_size = sensor_size
        self.n_time_bins = n_time_bins
        self.per_channel_normalization = per_channel_normalize
        self.dim = (self.n_time_bins, *self.sensor_size)

    def __call__(self, event_slices):
        """
        Apply the transformation to the given event slices.

        Args:
        - event_slices (Tensor): The input event slices.

        Returns:
        - Tensor: A batched tensor of voxel grids.
        """
        event_tensors = []
        for event_slice in event_slices:
            spike = self.to_event_spike(event_slice)
            event_tensors.append(spike)
        return np.array(event_tensors).astype(np.float32)



    def to_event_spike(self, events):
        assert "x" and "y" and "t" and "p" in events.dtype.names
        event_spike = np.zeros((3, self.n_time_bins, self.sensor_size[1], self.sensor_size[0]), np.float32).ravel()
        # normalize the event timestamps so that they lie between 0 and n_time_bins
        ts = (
            self.n_time_bins * (events["t"].astype(float) - np.min(events["t"])) / (np.max(events["t"]).astype(float) - np.min(events["t"]))
        )
        ts[ts == self.n_time_bins] = self.n_time_bins - 1

        xs = events["x"].astype(int)
        ys = events["y"].astype(int)
        pols = events["p"]


        pols[pols == 0] = -1 
        assert np.all((pols == -1) | (pols == 1))

        
        tis = ts.astype(int)

        assert np.all(tis < self.n_time_bins)
        assert np.all(tis >= 0)

        pos_indices = np.where(pols == 1)
        neg_indices = np.where(pols == -1)

        np.add.at(
            event_spike,
            xs[pos_indices] + ys[pos_indices] * self.sensor_size[0] + tis[pos_indices] * self.sensor_size[0] * self.sensor_size[1]
            + 0 * self.n_time_bins * self.sensor_size[0] * self.sensor_size[1],
            pols[pos_indices]
        )

        np.add.at(
            event_spike,
            xs[neg_indices] + ys[neg_indices] * self.sensor_size[0] + tis[neg_indices] * self.sensor_size[0] * self.sensor_size[1]
            + 1 * self.n_time_bins * self.sensor_size[0] * self.sensor_size[1],
            pols[neg_indices]
        )
        np.add.at(
            event_spike,
            xs[pos_indices] + ys[pos_indices] * self.sensor_size[0] + tis[pos_indices] * self.sensor_size[0] * self.sensor_size[1]
            + 2 * self.n_time_bins * self.sensor_size[0] * self.sensor_size[1],
            pols[pos_indices]
        )
        np.add.at(
            event_spike,
            xs[neg_indices] + ys[neg_indices] * self.sensor_size[0] + tis[neg_indices] * self.sensor_size[0] * self.sensor_size[1]
            + 2 * self.n_time_bins * self.sensor_size[0] * self.sensor_size[1],
            pols[neg_indices]
        )
    

        event_spike = np.reshape(
            event_spike, (3 * self.n_time_bins, self.sensor_size[1], self.sensor_size[0])
        )

        return event_spike






class SplitSequence:
    def __init__(self, sub_seq_length, stride):
        """
        Initialize the transformation.

        Args:
        - sub_seq_length (int): The length of each sub-sequence.
        - stride (int): The stride between sub-sequences.
        """
        self.sub_seq_length = sub_seq_length
        self.stride = stride

    def __call__(self, sequence, labels):
        """
        Apply the transformation to the given sequence and labels.

        Args:
        - sequence (Tensor): The input sequence of frames.
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of sub-sequences.
        - Tensor: A batched tensor of corresponding labels.
        """

        sub_sequences = []
        sub_labels = []

        for i in range(0, len(sequence) - self.sub_seq_length + 1, self.stride):
            sub_seq = sequence[i:i + self.sub_seq_length]
            sub_seq_labels = labels[i:i + self.sub_seq_length]
            sub_sequences.append(sub_seq)
            sub_labels.append(sub_seq_labels)

        return np.stack(sub_sequences), np.stack(sub_labels)


class SplitLabels:
    def __init__(self, sub_seq_length, stride):
        """
        Initialize the transformation.

        Args:
        - sub_seq_length (int): The length of each sub-sequence.
        - stride (int): The stride between sub-sequences.
        """
        self.sub_seq_length = sub_seq_length
        self.stride = stride
        # print(f"stride is {self.stride}")

    def __call__(self, labels):
        """
        Apply the transformation to the given sequence and labels.

        Args:
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of corresponding labels.
        """
        sub_labels = []

        for i in range(0, len(labels) - self.sub_seq_length + 1, self.stride):
            sub_seq_labels = labels[i:i + self.sub_seq_length]
            sub_labels.append(sub_seq_labels)

        return np.stack(sub_labels)

class ScaleLabel:
    def __init__(self, scaling_factor):
        """
        Initialize the transformation.

        Args:
        - scaling_factor (float): How much the spatial scaling was done on input
        """
        self.scaling_factor = scaling_factor


    def __call__(self, labels):
        """
        Apply the transformation to the given sequence and labels.

        Args:
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of corresponding labels.
        """
        labels[:,:2] =  labels[:,:2] * self.scaling_factor
        return labels

class TemporalSubsample:
    def __init__(self, temporal_subsample_factor):
        self.temp_subsample_factor = temporal_subsample_factor

    def __call__(self, labels):
        """
        temorally subsample the labels
        """
        interval = int(1/self.temp_subsample_factor)
        return labels[::interval]


class NormalizeLabel:
    def __init__(self, pseudo_width, pseudo_height):
        """
        Initialize the transformation.

        Args:
        - scaling_factor (float): How much the spatial scaling was done on input
        """
        self.pseudo_width = pseudo_width
        self.pseudo_height = pseudo_height

    def __call__(self, labels):
        """
        Apply normalization on label, with pseudo width and height

        Args:
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of corresponding labels.
        """
        labels[:, 0] = labels[:, 0] / self.pseudo_width
        labels[:, 1] = labels[:, 1] / self.pseudo_height
        return labels

@dataclass
class AugmentationState:
    apply_h_flip: bool
    apply_noise: bool
    apply_time_reversal: bool
    apply_random_time_shift: bool
    apply_crop: bool

def torch_uniform_sample_scalar(min_value: float, max_value: float):
    assert max_value >= min_value, f'{max_value=} is smaller than {min_value=}'
    if max_value == min_value:
        return min_value
    return min_value + (max_value - min_value) * th.rand(1).item()

class RandomSpatialAugmentor:
    def __init__(self,
                 dataset_wh: Tuple[int, int],
                 augm_config):

        def convert_to_namespace(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = convert_to_namespace(value)
            return argparse.Namespace(**d)

        augm_config = convert_to_namespace(augm_config)
        self.dataset_wh = dataset_wh

        self.h_flip_prob = augm_config.prob_hflip
        self.apply_noise_prob = augm_config.prob_noise
        self.apply_noise_max_factor = augm_config.max_noise_factor
        self.apply_noise_min_factor = augm_config.min_noise_factor
        self.apply_time_reversal = augm_config.time_reversal
        self.apply_crop = augm_config.random_crop
        self.apply_random_time_shift = augm_config.random_time_shift
        self.crop_factor = 0.8

        self.augm_state = AugmentationState(
            apply_h_flip=False,
            apply_noise=False,
            apply_time_reversal=False,
            apply_random_time_shift=False,
            apply_crop=False
            )

    def randomize_augmentation(self):
        self.augm_state.apply_h_flip = self.h_flip_prob > th.rand(1).item()
        self.augm_state.apply_noise = self.apply_noise_prob > th.rand(1).item()
        self.augm_state.apply_time_reversal = self.apply_time_reversal > th.rand(1).item()
        self.augm_state.apply_random_time_shift = self.apply_random_time_shift > th.rand(1).item()
        self.augm_state.apply_crop = self.apply_crop > th.rand(1).item()

    def h_flip(self, data):
        if len(data.shape) == 2:
            data[:, 0] = 1 -  data[:, 0]
            return data
        elif len(data.shape) == 4:
            return np.flip(data, axis=-1).copy()

    def add_random_noise(self, data):
        data_means = np.mean(data, axis=(1, 2, 3))
        data_stds = np.std(data, axis=(1, 2, 3))

        noise_factors = np.random.uniform(self.apply_noise_min_factor, self.apply_noise_max_factor, size=data.shape[0])
        noise = np.random.normal(loc=data_means[:, None, None, None], scale=data_stds[:, None, None, None] * noise_factors[:, None, None, None], size=data.shape).astype(data.dtype)

        noisy_data = data + noise
        return noisy_data

    def random_time_shift(self, input, target):
        start = np.random.randint(0, input.shape[0])
        return np.concatenate((input[start:], input[::-1]), axis=0)[:input.shape[0]], np.concatenate((target[start:], target[::-1]), axis=0)[:target.shape[0]]


    def random_crop_and_resize(self, input, target):
        """
        Randomly crop the input to 80% of its original size and resize it back,
        adjusting the target coordinates accordingly.

        Args:
            input (np.array): The input data with shape (N, C, H, W).
            target (np.array): The target coordinates with shape (N, 2), assuming (x, y) format.

        Returns:
            np.array: The resized input data to original dimensions.
            np.array: Adjusted target coordinates.
        """
        N, C, H, W = input.shape
        # crop factor randomly but max crop_factor
        crop = np.random.rand() * (1-self.crop_factor) + self.crop_factor
        crop_H = int(H * crop)
        crop_W = int(W * crop)

        # Calculate random start points for the crop
        start_H = np.random.randint(0, H - crop_H + 1)
        start_W = np.random.randint(0, W - crop_W + 1)

        cropped_and_resized_input = np.ones_like(input) * np.mean(input)
        adjusted_target = target.copy()
        for i in range(N):
            # Crop and resize for each frame
            cropped_frame = input[i, :, start_H:start_H+crop_H, start_W:start_W+crop_W]
            cropped_frame = cropped_frame.transpose(1, 2, 0)
            # resized_frame = cv2.resize(cropped_frame, (W,H), interpolation=cv2.INTER_LINEAR)
            # Assuming resized_frame is in (H, W, C) format; if it's a single-channel image, add a color dimension
            resized_frame = cropped_frame
            if resized_frame.ndim == 2:
                resized_frame = resized_frame[:, :, np.newaxis]
            

            resized_frame = resized_frame.transpose(2, 0, 1)

            cropped_and_resized_input[i,:, 0:crop_H, 0:crop_W] += resized_frame
            
            # Adjust target coordinates
            # Convert normalized coordinates to pixel space
            pixel_x = target[i, 0] * W
            pixel_y = target[i, 1] * H

            # Adjust for crop and resize
            adjusted_x = (pixel_x - start_W) / crop_W
            adjusted_y = (pixel_y - start_H) / crop_H

            # Ensure the adjusted coordinates are re-normalized to [0, 1] range
            adjusted_target[i, 0] = adjusted_x
            adjusted_target[i, 1] = adjusted_y

            # If any are outside we return the original back
            if adjusted_x < 0 or adjusted_y < 0 or adjusted_y > 1 or adjusted_x > 1:
                return input, target

        return cropped_and_resized_input, adjusted_target
    
    def __call__(self, input, target):
        self.randomize_augmentation()
        if self.augm_state.apply_h_flip:
            input =  self.h_flip(input)
            target = self.h_flip(target)
        if self.augm_state.apply_noise:
            input = self.add_random_noise(input)
        if self.augm_state.apply_time_reversal:
            input, target = input[::-1].copy(), target[::-1].copy()
        if self.augm_state.apply_random_time_shift:
            input, target = self.random_time_shift(input, target)
        if self.augm_state.apply_crop:
            input, target = self.random_crop_and_resize(input, target)
        return (input, target)
