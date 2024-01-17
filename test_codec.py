from data.dataloader import ImageDataset
import torch
from torchvision import transforms
from models.elic import TestModel as ELICModel
from models.fid import fid_pytorch, cal_psnr
from models.utils import print_avgs
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import pyiqa

ref_path = 'results_example_ffhq_elic_q1/elic/label'
dis_path = 'results_example_ffhq_elic_q1/elic/per_recon'

test_transforms = transforms.Compose(
    [transforms.ToTensor()]
)
ref_dataset = ImageDataset(root=ref_path, transform=test_transforms)
ref_dataloader = torch.utils.data.DataLoader(ref_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
ref_dataloader = list(ref_dataloader)
dis_dataset = ImageDataset(root=dis_path, transform=test_transforms)
dis_dataloader = torch.utils.data.DataLoader(dis_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
dis_dataloader = list(dis_dataloader)

fid_computer = fid_pytorch()

avgs = {
    "mse": [], "psnr": [],
    "fid": []
}

with torch.no_grad():
    fid_computer.clear_pools()
    for i, ((x, _), (x_hat, _)) in tqdm(enumerate(zip(ref_dataloader, dis_dataloader))):

        x1 = x.cuda()
        x2 = x_hat.cuda()

        unfold = nn.Unfold(kernel_size=(64, 64),stride=(64, 64))
        x1_unfold = unfold(x1).reshape(1, 3, 64, 64, -1)
        x1_unfold = torch.permute(x1_unfold, (0, 4, 1, 2, 3)).reshape(-1, 3, 64, 64)
        x2_unfold = unfold(x2).reshape(1, 3, 64, 64, -1)
        x2_unfold = torch.permute(x2_unfold, (0, 4, 1, 2, 3)).reshape(-1, 3, 64, 64)

        avgs['mse'].append(torch.mean((x1 - x2)**2).item())
        avgs['psnr'].append(cal_psnr(x1, x2))
        fid_computer.add_ref_img(x1_unfold)
        fid_computer.add_dis_img(x2_unfold)

    avgs['fid'].append(fid_computer.summary_pools())
    print_avgs(avgs)
