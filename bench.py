import argparse, json, os
import torch
from model.BaselineEyeTrackingModel import CNN_GRU
from model.RecurrentVisionTransformer import RVT
from utils.timer import CudaTimer


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
    model = eval(args.architecture)(args).to(args.device)
    factor = args.spatial_factor    

    # print number of params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Model has:", trainable_params, "trainable parameters")
    print("Model has:", non_trainable_params, "non-trainable parameters")
    data = torch.ones((1,1,3,int(640*factor), int(480*factor)))
    data = data.to(args.device)
    for i in range(1000):
        with CudaTimer(device=data.device, timer_name="model_inference"):
            output = model(data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_file", type=str, default='test_config.json', \
                        help="path to JSON configuration file")

    args = parser.parse_args()

    main(args)
