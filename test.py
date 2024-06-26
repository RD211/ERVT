"""
Author: Zuowen Wang
Affiliation: Insitute of Neuroinformatics, University of Zurich and ETH Zurich
Email: wangzu@ethz.ch
"""

import torch
import argparse, json, os, csv
from torch.utils.data import DataLoader
from dataset import ThreeETplus_Eyetracking, ScaleLabel, NormalizeLabel, \
    TemporalSubsample, NormalizeLabel, SliceLongEventsToShort, \
    EventSlicesToVoxelGrid, SliceByTimeEventsTargets
import tonic.transforms as transforms
from tonic import SlicedDataset
from model.RVT import RVT

def main(args):
    # Load hyperparameters from JSON configuration file
    if args.config_file:
        with open(os.path.join('./configs', args.config_file), 'r') as f:
            config = json.load(f)
        # Overwrite hyperparameters with command-line arguments
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
        args = argparse.Namespace(**config)
    else:
        raise ValueError("Please provide a JSON configuration file.")

    # Define your model, optimizer, and criterion
    model = eval(args.architecture)(args).to(args.device)

    # test data loader always cuts the event stream with the labeling frequency
    factor = args.spatial_factor
    temp_subsample_factor = args.temporal_subsample_factor

    label_transform = transforms.Compose([
        ScaleLabel(factor),
        TemporalSubsample(temp_subsample_factor),
        NormalizeLabel(pseudo_width=args.sensor_width*factor, pseudo_height=args.sensor_height*factor)
    ])

    test_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="test", \
                    transform=transforms.Downsample(spatial_factor=factor),
                    target_transform=label_transform)

    slicing_time_window = args.test_length*int(10000/temp_subsample_factor) #microseconds

    test_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=0, \
                    seq_length=args.test_length, seq_stride=args.test_stride, include_incomplete=True)

    post_slicer_transform = transforms.Compose([
        SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), overlap=0, include_incomplete=True),
        EventSlicesToVoxelGrid(sensor_size=(int(args.sensor_width*factor), int(args.sensor_height*factor), 2), \
                                    n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization)
    ])

    test_data = SlicedDataset(test_data_orig, test_slicer, transform=post_slicer_transform)

    assert args.batch_size == 1 
    # otherwise the collate function will through an error. 
    # This is only used in combination of include_incomplete=True during testing
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, \
                            num_workers=int(os.cpu_count()-2))

    # load weights from a checkpoint
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cuda', weights_only=False))
    else:
        raise ValueError("Please provide a checkpoint file.")
    
    # evaluate on the validation set and save the predictions into a csv file.
    model.eval()
    with open(args.output_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        # add column names 'row_id', 'x', 'y'
        csv_writer.writerow(['row_id', 'x', 'y'])
        row_id = 0
        for batch_idx, (data, target_placeholder) in enumerate(test_loader):
            data = data.to(args.device)

            output,_ = model(data)

            # Important! 
            # cast the output back to the downsampled sensor space (80x60)
            output = output * torch.tensor((args.sensor_width*factor, args.sensor_height*factor)).to(args.device)

            for sample in range(target_placeholder.shape[0]):
                for frame_id in range(target_placeholder.shape[1]):
                    row_to_write = output[sample][frame_id].tolist()
                    # prepend the row_id
                    row_to_write.insert(0, row_id)
                    csv_writer.writerow(row_to_write)
                    row_id += 1
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # a config file 
    parser.add_argument("--config_file", type=str, default='test_config', \
                        help="path to JSON configuration file")
    # load weights from a checkpoint
    parser.add_argument("--checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("--output_path", type=str, default='./submission.csv')

    args = parser.parse_args()

    main(args)
