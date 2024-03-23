import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from model.RVT import RVT
from utils.training_utils import train_epoch, validate_epoch, top_k_checkpoints
from utils.metrics import weighted_MSELoss
from dataset import ThreeETplus_Eyetracking, ScaleLabel, NormalizeLabel, TemporalSubsample, SliceLongEventsToShort, EventSlicesToVoxelGrid, SliceByTimeEventsTargets, RandomSpatialAugmentor
import tonic.transforms as transforms
from tonic import SlicedDataset, MemoryCachedDataset
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import tqdm
import matplotlib

def main(args):
    matplotlib.use('Agg')
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

    test_data = SlicedDataset(test_data_orig, val_slicer, transform=post_slicer_transform, metadata_path=f"./metadata/3et_test_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}")
    # cache the dataset to disk to speed up training. The first epoch will be slow, but the following epochs will be fast.
    val_data = MemoryCachedDataset(val_data)
    train_data = MemoryCachedDataset(train_data)
    test_data = MemoryCachedDataset(test_data)

    # Finally we wrap the dataset with pytorch dataloader
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, \
                                    num_workers=int(os.cpu_count()-2))
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, \
                                  num_workers=int(os.cpu_count()-2), pin_memory=True)
    
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, \
                                    num_workers=int(os.cpu_count()-2))
    
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
        def update(i):
            ax.clear()
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

    # We create 3 folders for train, val, and test if they don't exist
    os.makedirs(os.path.join(args.output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'test'), exist_ok=True)


    def wrap_in_html(html_str):
        return f"""
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        </head>
        <body>
        {html_str}
        </body>
        </html>
        """

    ############################################
    # We evaluate on train, val and test at the same time
    ############################################
    model.eval()
    for (i, (voxel_grid, target)), set in tqdm.tqdm([*zip(enumerate(train_loader), ["train"]*len(train_loader)), \
                                            *zip(enumerate(val_loader), ["val"]*len(val_loader)), \
                                            *zip(enumerate(test_loader), ["test"]*len(test_loader))], total=len(train_loader)+len(val_loader)+len(test_loader)):
        voxel_grid = voxel_grid.to(args.device)
        pred,_ = model(voxel_grid)
        pred = pred.detach().cpu().numpy().reshape(pred.shape[1], pred.shape[2])
        voxel_grid_np = voxel_grid[0, :, :, :, :].cpu().numpy()
        voxel_grid_np = (voxel_grid_np - voxel_grid_np.min()) / (voxel_grid_np.max() - voxel_grid_np.min())
        html_str = plot_voxel_grid_as_rgb_to_html(voxel_grid_np, f"Voxel grid {i}", pred, target[0][:,:2] if set != "test" else [])
        with open(os.path.join(args.output_path, set, f"{i}.html"), 'w') as f:
            f.write(wrap_in_html(html_str))
        
    ############################################
            


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # a config file 
    parser.add_argument("--config_file", type=str, help="path to JSON configuration file", required=True)
    parser.add_argument("--checkpoint", type=str, help="path to checkpoint", required=True)
    parser.add_argument("--output_path", type=str, default='web/')

    args = parser.parse_args()

    main(args)