import torch
import os
from utils.metrics import p_acc, p_acc_wo_closed_eye, px_euclidean_dist

def train_epoch(model, train_loader, criterion, optimizer, args):
    model.train()
    total_loss = 0.0
    total_p_corr_all = {f'p{p}_all':0 for p in args.pixel_tolerances}
    total_p_error_all  = {f'error_all':0}  # averaged euclidean distance
    total_samples_all, total_sample_p_error_all  = 0, 0


    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(args.device))
        # outputs = outputs.to(args.device) # why needed?
        #taking only the last frame's label, and first two dim are coordinate, last is open or close so discarded
        targets = targets.to(args.device)
        # Lt =Ls(Bk+1Bk+1)+Ls(BkBk)
        # L = Lr + Lt
        Lr = criterion(outputs[:, :, 0, :], targets[:,:, :2])
        Lt = 0
        if args.heads == 2:
            # roll the outputs tensor to the right by 1
            outputs_shifted = torch.roll(outputs, shifts=1, dims=1)
            outputs_shifted[:, 0, 1, :] = outputs[:, 0, 0, :] # B_1_' = B_0 (previous future pred at t=0 is current pred at t=0)
            Lt = criterion(outputs[:, :, 1, :], targets[:, :, 2:4]) + criterion(outputs[:, :, 0, :], outputs_shifted[:, :, 1, :])
        loss = Lr + Lt
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # calculate pixel tolerated accuracy
        p_corr, batch_size = p_acc(targets[:, :, :2], outputs[:, :, 0, :2], \
                                width_scale=args.sensor_width*args.spatial_factor, \
                                height_scale=args.sensor_height*args.spatial_factor, \
                                    pixel_tolerances=args.pixel_tolerances)
        total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in args.pixel_tolerances}
        total_samples_all += batch_size

        # calculate averaged euclidean distance
        p_error_total, bs_times_seqlen = px_euclidean_dist(targets[:, :, :2], outputs[:, :, 0, :2], \
                                width_scale=args.sensor_width*args.spatial_factor, \
                                height_scale=args.sensor_height*args.spatial_factor)
        total_p_error_all = {f'error_all': (total_p_error_all[f'error_all'] + p_error_total).item()}
        total_sample_p_error_all += bs_times_seqlen
    
    metrics = {'tr_p_acc_all': {f'tr_p{k}_acc_all': (total_p_corr_all[f'p{k}_all']/total_samples_all) for k in args.pixel_tolerances},
               'tr_p_error_all': {f'tr_p_error_all': (total_p_error_all[f'error_all']/total_sample_p_error_all)}}
    
    return model, total_loss / len(train_loader), metrics


def validate_epoch(model, val_loader, criterion, args):
    model.eval()
    total_loss = 0.0
    total_p_corr_all = {f'p{p}_all':0 for p in args.pixel_tolerances}
    total_p_error_all  = {f'error_all':0}
    total_samples_all, total_sample_p_error_all  = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs.to(args.device))
            targets = targets.to(args.device)
            loss = criterion(outputs[:, :, 0, :], targets[:,:, :2]) 
            total_loss += loss.item()

            # calculate pixel tolerated accuracy
            p_corr, batch_size = p_acc(targets[:, :, :2], outputs[:, :, 0, :], \
                                    width_scale=args.sensor_width*args.spatial_factor, \
                                    height_scale=args.sensor_height*args.spatial_factor, \
                                        pixel_tolerances=args.pixel_tolerances)
            total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in args.pixel_tolerances}
            total_samples_all += batch_size

            # calculate averaged euclidean distance
            p_error_total, bs_times_seqlen = px_euclidean_dist(targets[:, :, :2], outputs[:, :, 0, :], \
                                    width_scale=args.sensor_width*args.spatial_factor, \
                                    height_scale=args.sensor_height*args.spatial_factor)
            total_p_error_all = {f'error_all': (total_p_error_all[f'error_all'] + p_error_total).item()}
            total_sample_p_error_all += bs_times_seqlen

    metrics = {'val_p_acc_all': {f'val_p{k}_acc_all': (total_p_corr_all[f'p{k}_all']/total_samples_all) for k in args.pixel_tolerances},
                'val_p_error_all': {f'val_p_error_all': (total_p_error_all[f'error_all']/total_sample_p_error_all)}}
    
    return total_loss / len(val_loader), metrics


def top_k_checkpoints(args, artifact_uri):
    """
    only save the top k model checkpoints with the lowest validation loss.
    """
    # list all files ends with .pth in artifact_uri
    model_checkpoints = [f for f in os.listdir(artifact_uri) if f.endswith(".pth")]

    # but only save at most args.save_k_best models checkpoints
    if len(model_checkpoints) > args.save_k_best:
        # sort all model checkpoints by validation loss in ascending order
        model_checkpoints = sorted([f for f in os.listdir(artifact_uri) if f.startswith("model_best_ep")], \
                                    key=lambda x: float(x.split("_")[-1][:-4]))
        # delete the model checkpoint with the largest validation loss
        os.remove(os.path.join(artifact_uri, model_checkpoints[-1]))


