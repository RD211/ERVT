import argparse
import json
import os
import mlflow
import torch
from torch.utils.data import DataLoader
from model.BaselineEyeTrackingModel import CNN_GRU
from model.RecurrentVisionTransformer import RVT
from utils.training_utils import train_epoch, validate_epoch, top_k_checkpoints
from utils.metrics import weighted_MSELoss
from dataset import ThreeETplus_Eyetracking, ScaleLabel, NormalizeLabel, TemporalSubsample, SliceLongEventsToShort, EventSlicesToVoxelGrid, SliceByTimeEventsTargets, RandomSpatialAugmentor
import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import tqdm

# Parse arguments force insert the config
args = {
    "config_file": "./configs/rvt_2_layered_test.json",
    "checkpoint": "/users/mlica/eye/mlruns/431713750259427479/02262492d29142a0b6cb9dd0287eefd8/artifacts/model_best_ep133_val_loss_0.0090.pth",
    "train_length": 30,
    "val_length": 30,
    "train_stride": 15,
    "val_stride": 30,
    "data_augmentation": {
        "prob_hflip": 0.5,
        "prob_noise": 0.3,
        "max_noise_factor": 1.2,
        "min_noise_factor": 0.05,
    }
}

def main(args):

    # Load hyperparameters from JSON configuration file
    if args.config_file:
        with open(os.path.join('./configs', args.config_file), 'r') as f:
            config = json.load(f)
        # Overwrite command line arguments with config file
        for key, value in config.items():
            setattr(args, key, value)

    # Parameters from args (now includes config file parameters)
    factor = args.spatial_factor
    temp_subsample_factor = args.temporal_subsample_factor
    data_dir = args.data_dir
    n_time_bins = args.n_time_bins
    voxel_grid_ch_normaization = args.voxel_grid_ch_normaization

    label_transform = transforms.Compose([
        ScaleLabel(factor),
        TemporalSubsample(temp_subsample_factor),
        NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
    ])

    # Then we define the raw event recording and label dataset, the raw events spatial coordinates are also downsampled
    train_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="train", \
                    transform=transforms.Downsample(spatial_factor=factor), 
                    target_transform=label_transform)
    
    test_data_orig = ThreeETplus_Eyetracking(save_to=data_dir, split="test",
                    transform=transforms.Downsample(spatial_factor=factor),
                    target_transform=label_transform)

    slicing_time_window = args.train_length*int(10000/temp_subsample_factor)  # microseconds
    train_stride_time = int(10000/temp_subsample_factor*args.train_stride) #microseconds

    train_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window-train_stride_time, \
                    seq_length=args.train_length, seq_stride=args.train_stride, include_incomplete=False)
    

    post_slicer_transform = transforms.Compose([
        SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), overlap=0, include_incomplete=True),
        EventSlicesToVoxelGrid(sensor_size=(int(640*factor), int(480*factor), 2),
                            n_time_bins=n_time_bins, per_channel_normalize=voxel_grid_ch_normaization)
    ])

    train_data = SlicedDataset(train_data_orig, train_slicer, transform=post_slicer_transform, metadata_path=f"./metadata/3et_train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}")


    val_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="val", \
                            transform=transforms.Downsample(spatial_factor=factor),
                            target_transform=label_transform)

    # Then we slice the event recordings into sub-sequences. 
    # The time-window is determined by the sequence length (train_length, val_length) 
    # and the temporal subsample factor.
    slicing_time_window = args.train_length*int(10000/temp_subsample_factor) #microseconds

    # the validation set is sliced to non-overlapping sequences
    val_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=0, \
                    seq_length=args.val_length, seq_stride=args.val_stride, include_incomplete=False)

    # After slicing the raw event recordings into sub-sequences, 
    # we make each subsequences into your favorite event representation, 
    # in this case event voxel-grid
    post_slicer_transform = transforms.Compose([
        SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), overlap=0, include_incomplete=True),
        EventSlicesToVoxelGrid(sensor_size=(int(640*factor), int(480*factor), 2), \
                                    n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization)
    ])

    # We use the Tonic SlicedDataset class to handle the collation of the sub-sequences into batches.
    val_data = SlicedDataset(val_data_orig, val_slicer, transform=post_slicer_transform, metadata_path=f"./metadata/3et_val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}")

    augmentation = RandomSpatialAugmentor(dataset_wh = (1, 1), augm_config=args.data_augmentation) 

    # cache the dataset to disk to speed up training. The first epoch will be slow, but the following epochs will be fast.
    val_data = DiskCachedDataset(val_data, cache_path=f'./cached_dataset/val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}', transforms=augmentation)
    train_data = DiskCachedDataset(train_data, cache_path=f'./cached_dataset/train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}', transforms=augmentation)

    # Finally we wrap the dataset with pytorch dataloader
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, \
                                    num_workers=int(os.cpu_count()-2))
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, \
                                  num_workers=int(os.cpu_count()-2), pin_memory=True)
    model = eval(args.architecture)(args).to(args.device)

    # load weights from a checkpoint
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        raise ValueError("Please provide a checkpoint file.")
        


    def plot_voxel_grid_as_rgb_to_html(voxel_grid, title, predictions=[], targets=[]):
        voxel_grid = np.moveaxis(voxel_grid, 1, -1)  # N, C, H, W -> N, H, W, C
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(voxel_grid[0, :, :, :])

        def update(i):
            ax.clear()  # Clear to avoid overlaying dots
            ax.imshow(voxel_grid[i, :, :, :])
            ax.set_xticks([])
            ax.set_yticks([])
            if len(predictions) > 0:
                x, y = predictions[i]
                ax.plot(x*voxel_grid.shape[2], y*voxel_grid.shape[1], 'ro')
            if len(targets) > 0:
                x, y = targets[i]
                ax.plot(x*voxel_grid.shape[2], y*voxel_grid.shape[1], 'go')
            return ax,

        ani = FuncAnimation(fig, update, frames=range(voxel_grid.shape[0]), blit=False)
        html_str = ani.to_jshtml()
        plt.close(fig)
        return html_str

    # Initialize HTML document
    html_doc = """
    <html>
    <head>
    <title>Animation Gallery on Set """ + args.set + """</title>
    </head>
    <body>
    <h1>Animation Gallery on Set """ + args.set + """</h1>
    """

    # Assuming val_loader is defined and properly loaded
    for i, (voxel_grid, target) in tqdm.tqdm(enumerate(val_loader if args.set == "val" else train_loader)):
        voxel_grid = voxel_grid.to(args.device)
        pred = model(voxel_grid).detach().cpu().numpy()[0]
        voxel_grid_np = voxel_grid[0, :, :, :, :].cpu().numpy()
        voxel_grid_np = (voxel_grid_np - voxel_grid_np.min()) / (voxel_grid_np.max() - voxel_grid_np.min())  # Normalize
        html_str = plot_voxel_grid_as_rgb_to_html(voxel_grid_np, f"Voxel grid {i}", pred, target[0][:,:2])
        html_doc += html_str

    # Close HTML document
    html_doc += """
    </body>
    </html>
    """

    # Save the HTML document
    with open(args.output_path, 'w') as f:
        f.write(html_doc)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # a config file 
    parser.add_argument("--config_file", type=str, help="path to JSON configuration file", required=True)
    parser.add_argument("--checkpoint", type=str, help="path to checkpoint", required=True)
    parser.add_argument("--output_path", type=str, default='web/index.html')
    parser.add_argument("--set", type=str, default="val", help="Dataset split to visualize (val or test)")

    args = parser.parse_args()

    main(args)
