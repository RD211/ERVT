import json
import os
import torch
import torch.nn as nn
import numpy as np
import mlflow

def p_acc(target, prediction, width_scale, height_scale, pixel_tolerances=[1,3,5,10]):
    """
    Calculate the accuracy of prediction
    :param target: (N, seq_len, 2) tensor, seq_len could be 1
    :param prediction: (N, seq_len, 2) tensor
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 2)
    prediction = prediction.reshape(-1, 2)

    dis = target - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)

    total_correct = {}
    for p_tolerance in pixel_tolerances:
        total_correct[f'p{p_tolerance}'] = torch.sum(dist < p_tolerance)

    bs_times_seqlen = target.shape[0]
    return total_correct, bs_times_seqlen


def p_acc_wo_closed_eye(target, prediction, width_scale, height_scale, pixel_tolerances=[1,3,5,10]):
    """
    Calculate the accuracy of prediction, with p tolerance and only calculated on those with fully opened eyes
    :param target: (N, seqlen, 3) tensor
    :param prediction: (N, seqlen, 2) tensor, the last dimension is whether the eye is closed
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 3)
    prediction = prediction.reshape(-1, 2)

    dis = target[:,:2] - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)
    # check if there is nan in dist
    assert torch.sum(torch.isnan(dist)) == 0

    eye_closed = target[:,2] # 1 is closed eye
    # get the total number frames of those with fully opened eyes
    total_open_eye_frames = torch.sum(eye_closed == 0)

    # get the indices of those with closed eyes
    eye_closed_idx = torch.where(eye_closed == 1)[0]
    dist[eye_closed_idx] = np.inf
    total_correct = {}
    for p_tolerance in pixel_tolerances:
        total_correct[f'p{p_tolerance}'] = torch.sum(dist < p_tolerance)
        assert total_correct[f'p{p_tolerance}'] <= total_open_eye_frames

    return total_correct, total_open_eye_frames.item()


def px_euclidean_dist(target, prediction, width_scale, height_scale):
    """
    Calculate the total pixel euclidean distance between target and prediction
    in a batch over the sequence length
    :param target: (N, seqlen, 3) tensor
    :param prediction: (N, seqlen, 2) tensor
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 2)
    prediction = prediction.reshape(-1, 2)

    dis = target - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)

    total_px_euclidean_dist = torch.sum(dist)
    sample_numbers = target.shape[0]
    return total_px_euclidean_dist, sample_numbers


class weighted_MSELoss(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.weights = weights
        self.mseloss = nn.MSELoss(reduction='none')
        
    def forward(self, inputs, targets):
        batch_loss = self.mseloss(inputs, targets) * self.weights
        if self.reduction == 'mean':
            return torch.mean(batch_loss)
        elif self.reduction == 'sum':
            return torch.sum(batch_loss)
        else:
            return batch_loss
        
class weighted_RMSE(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super().__init__()
        self.weights = weights
        self.reduction = reduction
        self.mseloss = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets):
        batch_loss = self.mseloss(inputs, targets) * self.weights
        if self.reduction == 'mean':
            return torch.sqrt(torch.mean(batch_loss))
        elif self.reduction == 'sum':
            return torch.sqrt*(torch.sum(batch_loss))
        else:
            return batch_loss
        
def log_avg_metrics(train_metrics, val_metrics, args):
    mlflow.set_experiment(experiment_name=args.experiment_name)
    with mlflow.start_run(run_name=args.run_name + "_avg"):
        avg_val_metrics = {}
        avg_train_metrics = {}
        for epoch in range (args.num_epochs):
            for k in train_metrics[0][epoch].keys():
                avg_train_metrics[k] = sum_custom([train_metrics[i][epoch][k] for i in range(args.num_folds)])
            mlflow.log_metric("train_loss", avg_train_metrics["train_loss"], step=epoch)
            mlflow.log_metrics(avg_train_metrics['tr_p_acc_all'], step=epoch)
            mlflow.log_metrics(avg_train_metrics['tr_p_error_all'], step=epoch)
            if (epoch + 1) % args.val_interval == 0:	
                for k in val_metrics[0][(epoch + 1) // args.val_interval - 1].keys():
                    avg_val_metrics[k] = sum_custom([val_metrics[i][(epoch + 1) // args.val_interval - 1][k] for i in range(args.num_folds)])
                mlflow.log_metric("val_loss", avg_val_metrics["val_loss"], step=epoch)
                mlflow.log_metrics(avg_val_metrics['val_p_acc_all'], step=epoch)
                mlflow.log_metrics(avg_val_metrics['val_p_error_all'], step=epoch)

        # This fails on average
        with open(os.path.join(mlflow.get_artifact_uri(), "args.json"), 'w') as f:
            json.dump(vars(args), f)
        mlflow.log_artifact(os.path.join("./model", "BaselineEyeTrackingModel.py"))
        mlflow.log_artifact(os.path.join("./model", "RecurrentVisionTransformer.py"))
        mlflow.log_artifact(os.path.join("./model", "FastRecurrentTransformer.py"))
    mlflow.end_run()

def sum_custom(data):
    if type(data[0]) == dict:
        return {k: sum_custom([d[k] for d in data]) for k in data[0].keys()}
    else:
        return sum(data)/len(data)

