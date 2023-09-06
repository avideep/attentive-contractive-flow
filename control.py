import os
import sys
import argparse
import numpy as np

import torch

from torchvision.utils import save_image
import torch.nn as nn

sys.path.append(".")

from lib.anno import gen_images
from lib.resflow import ResidualFlow

parser = argparse.ArgumentParser(description='vae control')
parser.add_argument('--logdir', type=str, default="/users/gpu/avideep/normalizing-flows/attentive-resflow/residual-flows/experiments/celebahq64/attention-True")
parser.add_argument('--num', type=int, default=10)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--path', type=str, default="/users/gpu/avideep/normalizing-flows/attentive-resflow/residual-flows/experiments/celebahq64/attention-True/resflow-celeba64-attention-true-8-8-8-8-8.pth")
parser.add_argument('--attr', type=str, default="Eyeglasses")
args = parser.parse_args()

# Set GPU (Single GPU usage is only supported so far)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Log
logdir = os.path.join(args.logdir, "control")
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Model
enc_block_config_str = "128x1,128d2,128t64,64x3,64d2,64t32,32x3,32d2,32t16,16x7,16d2,16t8,8x3,8d2,8t4,4x3,4d4,4t1,1x2"
enc_channel_config_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:1024"

dec_block_config_str = "1x1,1u4,1t4,4x2,4u2,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
dec_channel_config_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:1024"

model = VAE(128, enc_block_config_str,
    dec_block_config_str,
    enc_channel_config_str,
    dec_channel_config_str,
).load_from_checkpoint(args.path, input_res = 128).to(device)

model.eval()

# Loader
gen = gen_images(args.attr, args.num)

with torch.no_grad():
    attr_z_list = []
    for i, (p, n) in enumerate(gen):
        p = torch.unsqueeze(p.to(device), 0)
        n = torch.unsqueeze(n.to(device), 0)

        p_mu, p_logvar = model.encode(p)
        p_z = model.reparameterize(p_mu, p_logvar)
        n_mu, n_logvar = model.encode(n)
        n_z = model.reparameterize(n_mu, n_logvar)

        del p_mu, p_logvar, n_mu, n_logvar
        attr_z_list.append(p_z - n_z)
    attr_zs = torch.rand(args.num, 1024, 1, 1).to(device)
    torch.cat(attr_z_list, out = attr_zs)
    del attr_z_list
    attr_z = torch.mean(attr_zs, dim = 0).view(1,1024,1,1)
    for i, (p,n) in enumerate(gen):
        n = torch.unsqueeze(n.to(device), 0)
        n_mu, n_logvar = model.encode(n)
        n_z = model.reparameterize(n_mu, n_logvar)
        del n_mu, n_logvar
        break
    p_z = n_z + attr_z 
    recons_p = model.decode(p_z)
    recons_n = model.decode(n_z)
    image = torch.zeros(2,3,128,128)
    image[0,:, :, :] = recons_n
    image[1,:, :, :] = recons_p
    save_image(image,logdir + "/image_"+ args.attr + ".png")



        # Interpolate between p_z and n_z
        # rec_tensors = torch.zeros((10, 3, 128, 128))

        # for j, alpha in enumerate(np.arange(0.0, 1.0, 0.1)):
        #     z_inp = p_z - alpha * (p_z - n_z) 
        #     rec_tensors[j, :, :, :] = model.decode(z_inp)

        # save_image(rec_tensors,logdir + "/image_"+ str(i+1) + ".png")


