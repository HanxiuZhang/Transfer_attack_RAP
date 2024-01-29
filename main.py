import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torchvision
import torchvision.datasets as td
import torch.distributions as tdist
from torchvision import models, transforms
from PIL import Image
import csv
import numpy as np
import os
import scipy.stats as st

from utils import *

arg = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.device)
arg.adv_alpha = arg.adv_epsilon / arg.adv_steps

exp_name = arg.source_model + '_' + arg.loss_function + '_'

if arg.targeted:
    exp_name += 'T_'
if arg.MI:
    exp_name += 'MI_'
if arg.DI:
    exp_name += 'DI_'
if arg.TI:
    exp_name += 'TI_'
if arg.SI:
    exp_name += 'SI_'
if arg.m1 != 1:
    exp_name += f'm1_{arg.m1}_'
if arg.m2 != 1:
    exp_name += f'm2_{arg.m2}_'
if arg.strength != 0:
    exp_name += 'Admix_'


exp_name += str(arg.transpoint)


if arg.targeted:
    exp_name += '_target'


# for targeted attack, we need to conduct the untargeted attack during the inner loop.
# for untargeted attack, we need to conduct the targeted attack (the true label) during the inner loop. 
if not arg.targeted:
    arg.adv_targeted = 1
else:
    arg.adv_targeted = 0


if arg.save:
    arg.file_path = "./save/"+exp_name
    makedir(arg.file_path)


logging(exp_name.format())
logging('Hyper-parameters: {}\n'.format(arg.__dict__))


model_1 = models.inception_v3(pretrained=True, transform_input=True).eval()
model_2 = models.resnet50(pretrained=True).eval()
model_3 = models.densenet121(pretrained=True).eval()
model_4 = models.vgg16_bn(pretrained=True).eval()


for param in model_1.parameters():
    param.requires_grad = False
for param in model_2.parameters():
    param.requires_grad = False
for param in model_3.parameters():
    param.requires_grad = False
for param in model_4.parameters():
    param.requires_grad = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging(f'device: {device}')

model_1.to(device)
model_2.to(device)
model_3.to(device)
model_4.to(device)

if arg.source_model == 'inception-v3':
    model_source = model_1
elif arg.source_model == 'resnet50':
    model_source = model_2
elif arg.source_model == 'densenet121':
    model_source = model_3
elif arg.source_model == 'vgg16bn':
    model_source = model_4

logging("setting up the source and target models")

torch.manual_seed(arg.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(arg.seed)


# values are standard normalization for ImageNet images,
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py

trn = transforms.Compose([transforms.ToTensor(), ])
image_id_list, label_ori_list, label_tar_list = load_ground_truth('/home/hancy/dataset/img1k/images.csv')

img_size = 299
input_path = '/home/hancy/dataset/img1k/images/'
lr = 2 / 255  # step size
epsilon = 16  # L_inf norm bound
num_batches = int(np.ceil(len(image_id_list) / arg.batch_size))

logging("loaded the images".format())
n = tdist.Normal(0.0, 15/255)

#-------------------------------------#


logging("finishing the attack experiment")
logging(50*"#")
logging(50*"#")
logging(50*"#")

