from data.dataloader import ImageDataset
import torch
from torchvision import transforms
from models.elic import TestModel as ELICModel
from models.fid import fid_pytorch, cal_psnr
from models.utils import print_avgs
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from guided_diffusion.measurements import elic_paths

lambdas = [1,2,3,4,5]

for lam in lambdas:
    model_path = elic_paths[lam - 1]
    test_path = 'data/ffhq_samples/'
    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )
    test_dataset = ImageDataset(root=test_path, transform=test_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    codec = ELICModel()
    codec.load_state_dict(torch.load(model_path))
    codec = codec.cuda()
    codec.eval()
    fid_computer = fid_pytorch()
    avgs = {
        "bpp": [], "y_bpp": [], "z_bpp": [],
        "mse": [], "psnr": [],
        "fid": []
    }
    with torch.no_grad():
        fid_computer.clear_pools()
        for i, (x, x_name) in tqdm(enumerate(test_dataloader)):
            x = x.cuda()
            b, c, h, w = x.shape
            num_pix = h*w
            # encode
            enc_out = codec(x, "enc", False)
            y_hat = enc_out["y_hat"]
            y_bpp = torch.mean(torch.sum(-torch.log2(enc_out["likelihoods"]["y"]),dim=(1,2,3)), dim=0) / num_pix
            z_bpp = torch.mean(torch.sum(-torch.log2(enc_out["likelihoods"]["z"]),dim=(1,2,3)), dim=0) / num_pix
            # decode
            dec_out = codec(y_hat, "dec", False)
            x_hat = dec_out["x_bar"]
            unfold = nn.Unfold(kernel_size=(64, 64),stride=(64, 64))
            x1_unfold = unfold(x).reshape(1, 3, 64, 64, -1)
            x1_unfold = torch.permute(x1_unfold, (0, 4, 1, 2, 3)).reshape(-1, 3, 64, 64)
            x2_unfold = unfold(x_hat).reshape(1, 3, 64, 64, -1)
            x2_unfold = torch.permute(x2_unfold, (0, 4, 1, 2, 3)).reshape(-1, 3, 64, 64)
            fid_computer.add_ref_img(x1_unfold)
            fid_computer.add_dis_img(x2_unfold)
            # statistics
            avgs['bpp'].append(y_bpp.item() + z_bpp.item())
            avgs['y_bpp'].append(y_bpp.item())
            avgs['z_bpp'].append(z_bpp.item())
            avgs['mse'].append(torch.mean((x - x_hat)**2).item())
            avgs['psnr'].append(cal_psnr(x, x_hat))
            # compute fid in 256 patches to ensure enough images is available
            # flush results if being told so
        avgs['fid'].append(fid_computer.summary_pools())
        print_avgs(avgs)
