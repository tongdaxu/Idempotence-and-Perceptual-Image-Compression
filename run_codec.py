from functools import partial
import os

import argparse
import yaml

import torch

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color
from util.logger import get_logger

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
       
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    logger.info(f"Operation: {measure_config['operator']['name']}")

    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['label', 'mse_recon', 'progress', 'per_recon', 're_recon']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    init_scale = task_config['conditioning']['params']['scale']
    maxrepeat = task_config['conditioning']['params']['maxrepeat']
    mse_mses, per_mses = [], []
    # Do Inference
    for i, ref_img in enumerate(loader):
        repeat = 1
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        y_n = operator.forward(ref_img)
         
        # Sampling
        dist = 1e6
        cur_scale = init_scale

        while dist > 20 and repeat <= maxrepeat:
            x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
            # Prepare conditioning method
            cond_config = task_config['conditioning']
            cond_config['params']['scale'] = cur_scale
            cond_method = get_conditioning_method(cond_config['method'], operator, None, **cond_config['params'])
            measurement_cond_fn = cond_method.conditioning        
            # Load diffusion sampler
            sampler = create_sampler(**diffusion_config) 
            sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
            sample, dist = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)
            repeat += 1
            cur_scale *= 1.5

        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'mse_recon', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'per_recon', fname), clear_color(sample))
        plt.imsave(os.path.join(out_path, 're_recon', fname), clear_color(operator.forward(sample)))
        y_n = (y_n + 1.0) / 2.0
        ref_img = (ref_img + 1.0) / 2.0
        sample = (sample + 1.0) / 2.0
        mse_mse = torch.mean((ref_img - y_n)**2)
        per_mse = torch.mean((ref_img - sample)**2)
        mse_mses.append(mse_mse.item())
        per_mses.append(per_mse.item())
        print("original mse: {0:.4} / {1:.4}, perceptual mse: {2:.4} / {3:.4} x {4} times with scale {5:.3}".format(mse_mse, np.mean(mse_mses), per_mse, np.mean(per_mses), repeat - 1, cur_scale / 1.5))

if __name__ == '__main__':
    main()
