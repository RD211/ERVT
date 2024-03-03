"""
This script implements a PyTorch deep learning training pipeline for an eye tracking application.
It includes a main function to pass in arguments, train and validation functions, and uses MLflow as the logging library.
The script also supports fine-grained deep learning hyperparameter tuning using argparse and JSON configuration files.
All hyperparameters are logged with MLflow.

Author: Zuowen Wang
Affiliation: Insitute of Neuroinformatics, University of Zurich and ETH Zurich
Email: wangzu@ethz.ch
"""

import argparse, json, os, mlflow
import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.BaselineEyeTrackingModel import CNN_GRU
from model.RecurrentVisionTransformer import RVT
from model.FastRecurrentTransformer import FRT
from utils.training_utils import set_deterministic, train_epoch, validate_epoch, top_k_checkpoints
from utils.metrics import log_avg_metrics, weighted_MSELoss, weighted_RMSE
from dataset import ThreeETplus_Eyetracking, ScaleLabel, NormalizeLabel, \
    TemporalSubsample, NormalizeLabel, SliceLongEventsToShort, \
    EventSlicesToVoxelGrid, SliceByTimeEventsTargets, RandomSpatialAugmentor, EventSlicesToSpikeTensor
import tonic.transforms as transforms
from tonic import SlicedDataset, MemoryCachedDataset
from sklearn.model_selection import KFold

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, args, fold_num):
    best_val_loss = float("inf")
    metrics_return_train = []
    metrics_return_val = []
    # Training loop
    try:
        for epoch in range(args.num_epochs):
            model, train_loss, metrics = train_epoch(model, train_loader, criterion, optimizer, args)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metrics(metrics['tr_p_acc_all'], step=epoch)
            mlflow.log_metrics(metrics['tr_p_error_all'], step=epoch)
            metrics["train_loss"] = train_loss
            metrics_return_train.append(metrics)

            if args.val_interval > 0 and (epoch + 1) % args.val_interval == 0:
                val_loss, val_metrics = validate_epoch(model, val_loader, criterion, args)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # save the new best model to MLflow artifact with 3 decimal places of validation loss in the file name
                    torch.save(model.state_dict(), os.path.join(mlflow.get_artifact_uri(), \
                                f"model_best_ep{epoch}_val_loss_{val_loss:.4f}.pth"))

                    # DANGER Zone, this will delete files (checkpoints) in MLflow artifact
                    top_k_checkpoints(args, mlflow.get_artifact_uri())

                print(f"[Validation] at Epoch {epoch+1}/{args.num_epochs}: Val Loss: {val_loss:.4f}")
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metrics(val_metrics['val_p_acc_all'], step=epoch)
                mlflow.log_metrics(val_metrics['val_p_error_all'], step=epoch)
                val_metrics["val_loss"] = val_loss
                metrics_return_val.append(val_metrics)
            # Print progress
            print(f"Epoch {epoch+1}/{args.num_epochs}: Train Loss: {train_loss:.4f} on fold {fold_num+1}/{args.num_folds}")
            scheduler.step()
    except Exception as e:
        print("Exception occurred during training, terminating training process on this fold.", e)

    return model, metrics_return_train, metrics_return_val


def process_fold(fold, train_index, val_index, args, data, temp_subsample_factor, factor):
     # This needs to include necessary imports and definitions that are used inside this function
    mlflow.set_tracking_uri(args.mlflow_path)
    mlflow.set_experiment(experiment_name=args.experiment_name)

    
    train_data_orig = torch.utils.data.Subset(data, train_index)
    val_data_orig = torch.utils.data.Subset(data, val_index)

    # Then we slice the event recordings into sub-sequences.
    # The time-window is determined by the sequence length (train_length, val_length)
    # and the temporal subsample factor.
    slicing_time_window = args.train_length*int(10000/temp_subsample_factor) #microseconds
    train_stride_time = int(10000/temp_subsample_factor*args.train_stride) #microseconds

    train_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window-train_stride_time, \
                    seq_length=args.train_length, seq_stride=args.train_stride, include_incomplete=False)
    # the validation set is sliced to non-overlapping sequences
    val_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=0, \
                    seq_length=args.val_length, seq_stride=args.val_stride, include_incomplete=False)

    # After slicing the raw event recordings into sub-sequences,
    # we make each subsequences into your favorite event representation,
    # in this case event voxel-grid
    post_slicer_transform = transforms.Compose([
        SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), overlap=0, include_incomplete=True),
        EventSlicesToSpikeTensor(sensor_size=(int(640*factor), int(480*factor), 2), \
                                n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization)
    ])

    # We use the Tonic SlicedDataset class to handle the collation of the sub-sequences into batches.
    train_data = SlicedDataset(train_data_orig, train_slicer, transform=post_slicer_transform, metadata_path=f"./metadata/3et_train_tl_{fold}_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}")
    val_data = SlicedDataset(val_data_orig, val_slicer, transform=post_slicer_transform, metadata_path=f"./metadata/3et_val_vl_{fold}_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}")

    print(args.run_name + f"_fold{fold+1} / {args.num_folds}")
    augmentation = RandomSpatialAugmentor(dataset_wh = (1, 1), augm_config=args.data_augmentation)
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "weighted_mse":
        criterion = weighted_MSELoss(weights=torch.tensor((args.sensor_width/args.sensor_height, 1)).to(args.device), \
                                        reduction='mean')
    elif args.loss == "weighted_rmse":
        criterion = weighted_RMSE(weights=torch.tensor((args.sensor_width/args.sensor_height, 1)).to(args.device), \
                                        reduction='mean')
    else:
        raise ValueError("Invalid loss name")
    
    train_data = MemoryCachedDataset(train_data, transforms=augmentation)
    val_data = MemoryCachedDataset(val_data)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    with mlflow.start_run(run_name=args.run_name + f"_fold{fold+1} / {args.num_folds}"):
        # ====== MLflow logging ======
        mlflow.log_artifact(__file__)
        mlflow.log_params(vars(args))
        with open(os.path.join(mlflow.get_artifact_uri(), "args.json"), 'w') as f:
            json.dump(vars(args), f)
        for file in os.listdir("./model"):
            if file.endswith(".py"):
                mlflow.log_artifact(os.path.join("./model", file))
        # ============================
                    
        model = eval(args.architecture)(args).to(args.device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma, -1)

        model, train_m, val_m = train(model, train_loader, val_loader, criterion, optimizer, scheduler, args, fold)
            
        torch.save(model.state_dict(), os.path.join(mlflow.get_artifact_uri(), f"model_fold{fold+1}.pth"))
        mlflow.end_run()
            
        return train_m, val_m
        
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
    
    set_deterministic(args.seed)

    factor = args.spatial_factor # spatial downsample factor
    temp_subsample_factor = args.temporal_subsample_factor # downsampling original 100Hz label to 20Hz
    # First we define the label transformations
    label_transform = transforms.Compose([
        ScaleLabel(factor),
        TemporalSubsample(temp_subsample_factor),
        NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
    ])
    # Then we define the raw event recording and label dataset, the raw events spatial coordinates are also downsampled
    train_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="train", \
                    transform=transforms.Downsample(spatial_factor=factor),
                    target_transform=label_transform)
    val_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="val", \
                    transform=transforms.Downsample(spatial_factor=factor),
                    target_transform=label_transform)

    data = torch.utils.data.ConcatDataset([train_data_orig, val_data_orig])
    
    def run_in_parallel(args, data):
        multiprocessing.set_start_method('spawn', force=True)
        kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.random_state)
        folds = list(kf.split(data))
        train_metrics = []
        val_metrics = []
        args_dict = args
        with ProcessPoolExecutor() as executor:
            results = executor.map(process_fold, *zip(*[(fold, train_index, val_index, args_dict, copy.deepcopy(data), temp_subsample_factor, factor
                                                         ) for fold, (train_index, val_index) in enumerate(folds)]))
        for train_m, val_m in results:
            train_metrics.append(train_m)
            val_metrics.append(val_m)

        log_avg_metrics(train_metrics, val_metrics, args)
    multiprocessing.set_start_method('spawn', force=True)
    run_in_parallel(args, data)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # training management arguments
    parser.add_argument("--mlflow_path", type=str, help="path to MLflow tracking server")
    parser.add_argument("--experiment_name", type=str, help="name of the experiment")
    parser.add_argument("--run_name", type=str, help="name of the run")

    # a config file
    parser.add_argument("--config_file", type=str, default=None, help="path to JSON configuration file")

    # training hyperparameters
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--num_epochs", type=int, help="number of epochs")

    args = parser.parse_args()

    main(args)
