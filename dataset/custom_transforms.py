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
class ZoomOutState:
    active: bool
    x0: int
    y0: int
    zoom_out_factor: float


@dataclass
class RotationState:
    active: bool
    angle_deg: float


@dataclass
class AugmentationState:
    apply_h_flip: bool
    rotation: RotationState
    apply_zoom_in: bool
    zoom_out: ZoomOutState

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

        augm_config = convert_to_namespace(augm_config['random'])
        self.dataset_wh = dataset_wh

        self.h_flip_prob = augm_config.prob_hflip

        self.augm_state = AugmentationState(
            apply_h_flip=False,
            rotation=RotationState(active=False, angle_deg=0.0),
            apply_zoom_in=False,
            zoom_out=ZoomOutState(active=False, x0=0, y0=0, zoom_out_factor=1.0))

    def randomize_augmentation(self):
        self.augm_state.apply_h_flip = self.h_flip_prob > th.rand(1).item()

    def h_flip(self, data):
        if len(data.shape) == 2:
            data[:, 0] = 1 -  data[:, 0]
            return data
        elif len(data.shape) == 4:
            return np.flip(data, axis=-1).copy()
       

    def __call__(self, input, target):
        self.randomize_augmentation()
        if self.augm_state.apply_h_flip:
            return self.h_flip(input), self.h_flip(target)

        return (input, target)