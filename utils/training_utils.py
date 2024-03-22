import random
import numpy as np
import torch
import os
from utils.metrics import p_acc, px_euclidean_dist

def train_epoch(model, train_loader, criterion, optimizer, args):
    model.train()
    total_loss = 0.0
    total_p_corr_all = {f'p{p}_all':0 for p in args.pixel_tolerances}
    total_p_error_all  = {f'error_all':0}  # averaged euclidean distance
    total_samples_all, total_sample_p_error_all  = 0, 0

    # NOTE: the optimization strategy is TBPTT!
    # the Pytorch model needs to accept and return the hidden states of the RNN
    # in order to detach the gradients at specific intervals
    # if args.tbptt is equal to args.train_length, this should be equivalent to BPTT

    assert args.train_length % args.tbptt == 0, "The sequence length has to be divisible by the TBPTT split"
    chuncks = [args.tbptt] * (args.train_length // args.tbptt)
    for inputs, targets in train_loader:
        # the batched input needs to be split in equally sized chuncks.
        # normal BPTT is applied to every chuck, with the hidden states shared between chuncks.
        split_inputs = torch.split(inputs, chuncks, dim=1)
        split_targets = torch.split(targets, chuncks, dim=1)

        hidden = None # the initial hidden states, this is equivalent to lstm_states
        seq_loss = 0 # sequence loss accumulated over all chuncks
        optimizer.zero_grad()
        acc_outputs = []
        for x, y in zip(split_inputs, split_targets):
            outputs, hidden = model(x.to(args.device), hidden)
            y = y.to(args.device)
            loss = criterion(outputs, y[:, :, :2])
            seq_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # accumulate outputs to recreate the whole sequence
            acc_outputs.append(outputs)

            # avoid passing gradients further, detach the hidden states from the computational graph
            # NOTE: hidden is considered to be a list (corresponding to multiple stages)
            for i in range(len(hidden)):
                hidden[i] = (hidden[i][0].detach(), hidden[i][1].detach())

        outputs = torch.cat(acc_outputs, dim=1).detach().cpu() # concatenate over the time dimension
        total_loss += seq_loss / len(chuncks) # track the whole sequence loss
        # calculate pixel tolerated accuracy
        p_corr, batch_size = p_acc(targets[:, :, :2], outputs[:, :, :], \
                                width_scale=args.sensor_width*args.spatial_factor, \
                                height_scale=args.sensor_height*args.spatial_factor, \
                                    pixel_tolerances=args.pixel_tolerances)
        total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in args.pixel_tolerances}
        total_samples_all += batch_size

        # calculate averaged euclidean distance
        p_error_total, bs_times_seqlen = px_euclidean_dist(targets[:, :, :2], outputs[:, :, :], \
                                width_scale=args.sensor_width*args.spatial_factor, \
                                height_scale=args.sensor_height*args.spatial_factor)
        total_p_error_all = {f'error_all': (total_p_error_all[f'error_all'] + p_error_total).item()}
        total_sample_p_error_all += bs_times_seqlen

    metrics = {'tr_p_acc_all': {f'tr_p{k}_acc_all': (total_p_corr_all[f'p{k}_all']/total_samples_all) for k in args.pixel_tolerances},
               'tr_p_error_all': {f'tr_p_error_all': (total_p_error_all[f'error_all']/total_sample_p_error_all)}}

    return model, total_loss / len(train_loader), metrics


def validate_epoch(model, val_loader, criterion, args):
    # NOTE: the validation of the model should not be influenced by using a different optimization procedure (TBPTT)
    model.eval()
    total_loss = 0.0
    total_p_corr_all = {f'p{p}_all':0 for p in args.pixel_tolerances}
    total_p_error_all  = {f'error_all':0}
    total_samples_all, total_sample_p_error_all  = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs, _ = model(inputs.to(args.device))
            targets = targets.to(args.device)
            loss = criterion(outputs, targets[:,:, :2])
            total_loss += loss.item()

            # calculate pixel tolerated accuracy
            p_corr, batch_size = p_acc(targets[:, :, :2], outputs[:, :, :], \
                                    width_scale=args.sensor_width*args.spatial_factor, \
                                    height_scale=args.sensor_height*args.spatial_factor, \
                                        pixel_tolerances=args.pixel_tolerances)
            total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in args.pixel_tolerances}
            total_samples_all += batch_size

            # calculate averaged euclidean distance
            p_error_total, bs_times_seqlen = px_euclidean_dist(targets[:, :, :2], outputs[:, :, :], \
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
                                    key=lambda x: -float(x.split("_")[-1][:-4]))
        # delete the model checkpoint with the largest validation loss
        os.remove(os.path.join(artifact_uri, model_checkpoints[-1]))