################################################################################################
# This script is used to benchmark the model. It loads the model from the configuration file
# and runs the model on a dummy input tensor to get the model summary and inference time.
################################################################################################

import torch
import argparse, json, os
from utils.timer import CudaTimer
from torchinfo import summary
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
    model = eval(args.architecture)(args).to(args.device)

    # Create a dummy input tensor
    factor = args.spatial_factor    
    data = torch.ones((1,1,3,int(args.sensor_width*factor), int(args.sensor_height*factor)))
    data = data.to(args.device)

    # print model summary
    summary(model, input_data=data, verbose=2)
    
    # We need to set the model to evaluation mode and compile the model before benchmarking
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.set_float32_matmul_precision('medium')
    model.eval()
    model = torch.compile(model)

    with torch.no_grad():
        for _ in range(1000):
            with CudaTimer(device=data.device, timer_name="model_inference"):
                output = model(data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_file", type=str, default='test_config.json', \
                        help="path to JSON configuration file")

    args = parser.parse_args()

    main(args)
