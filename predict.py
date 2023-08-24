# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import os
import yaml
import math
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_efficient, ft_net_NAS, \
    ft_net_convnext, PCB, PCB_test
from utils import fuse_all_conv_bn
from shutil import copyfile
from PIL import Image
import torch.nn.functional as F

try:
    from apex.fp16_utils import *
except ImportError:  # will be 3.x series
    print(
        'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--predict_dir', default='../Market/pytorch', type=str, help='./predict_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_efficient', action='store_true', help='use efficient-b4')
parser.add_argument('--use_hr', action='store_true', help='use hr18 net')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--multi', action='store_true', help='use multiple query')
parser.add_argument('--fp16', action='store_true', help='use fp16.')
parser.add_argument('--ibn', action='store_true', help='use ibn.')
parser.add_argument('--ms', default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--save', default='true', type=str, help='save result')
parser.add_argument('--level', default='0', type=str, help='level: e.g. 0,1,2,3,4')
parser.add_argument('--save_dir', default='../save', type=str, help='./save_result')
opt = parser.parse_args()

# clear directorys
# os.system("rm -f /root/Person_reID/datasets/save/02/*.*")
# os.system("rm -f /root/Person_reID/datasets/save/01/*.*")

###load config###
# load the training config
config_path = os.path.join('./model', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)  # for the new pyyaml via 'conda install pyyaml'
opt.fp16 = config['fp16']
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
if 'use_swin' in config:
    opt.use_swin = config['use_swin']
if 'use_swinv2' in config:
    opt.use_swinv2 = config['use_swinv2']
if 'use_convnext' in config:
    opt.use_convnext = config['use_convnext']
if 'use_efficient' in config:
    opt.use_efficient = config['use_efficient']
if 'use_hr' in config:
    opt.use_hr = config['use_hr']

opt.nclasses = 2

if 'ibn' in config:
    opt.ibn = config['ibn']
if 'linear_num' in config:
    opt.linear_num = config['linear_num']

str_ids = opt.gpu_ids.split(',')

# which_epoch = opt.which_epoch
name = opt.name
save = 1
if opt.save == "false":
    save = 0
predict_dir = opt.predict_dir
save_dir = opt.save_dir
level = opt.level
# 默认状态
if level == "0":
    thre1 = 0.5
    thre2 = 0.5
# 分类结果略微偏向合规，漏检更多，误检更少
elif level == "1":
    thre1 = 0.5
    thre2 = 0.7
# 分类结果更加偏向合规，漏检更多，误检更少
elif level == "2":
    thre1 = 0.5
    thre2 = 0.9
# 分类结果略微偏向超标，漏检更少，误检更多
elif level == "3":
    thre1 = 0.7
    thre2 = 0.5
# 分类结果更加偏向超标，漏检更少，误检更多
elif level == "4":
    thre1 = 0.9
    thre2 = 0.5
else:
    print("Illegal level input")

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)
print('We use the scale: %s' % opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

#########################################################################
# Load Data                                                             #
# ---------                                                             #   
#                                                                       #
# We will use torchvision and torch.utils.data packages for loading the #
# data.                                                                 #
#########################################################################
if opt.use_swin:
    h, w = 224, 224
else:
    h, w = 256, 128

data_transforms = transforms.Compose([
    transforms.Resize((h, w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384, 192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    h, w = 384, 192

data_dir = predict_dir
use_gpu = torch.cuda.is_available()


######################################################################
# Load model
######################################################################

def load_network(network):
    save_path = os.path.join('./model', name, 'net_%s.pth' % opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
######################################################################
# Load Collected data Trained model
print('-------predict-----------')
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses, stride=opt.stride, linear_num=opt.linear_num)
elif opt.use_NAS:
    model_structure = ft_net_NAS(opt.nclasses, linear_num=opt.linear_num)
elif opt.use_swin:
    model_structure = ft_net_swin(opt.nclasses, linear_num=opt.linear_num)
elif opt.use_swinv2:
    model_structure = ft_net_swinv2(opt.nclasses, (h, w), linear_num=opt.linear_num)
elif opt.use_convnext:
    model_structure = ft_net_convnext(opt.nclasses, linear_num=opt.linear_num)
elif opt.use_efficient:
    model_structure = ft_net_efficient(opt.nclasses, linear_num=opt.linear_num)
elif opt.use_hr:
    model_structure = ft_net_hr(opt.nclasses, linear_num=opt.linear_num)
else:
    model_structure = ft_net(opt.nclasses, stride=opt.stride, ibn=opt.ibn, linear_num=opt.linear_num)

if opt.PCB:
    model_structure = PCB(opt.nclasses)

model = load_network(model_structure)

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# print('Here I fuse conv and bn for faster inference, and it does not work for transformers. Comment out this following line if you do not want to fuse conv&bn.')
model = fuse_all_conv_bn(model)

# We can optionally trace the forward method with PyTorch JIT so it runs faster.
# To do so, we can call `.trace` on the reparamtrized module with dummy inputs
# expected by the module.
# Comment out this following line if you do not want to trace.
dummy_forward_input = torch.rand(opt.batchsize, 3, h, w).cuda()
model = torch.jit.trace(model, dummy_forward_input)

img_path = data_dir
save_path = save_dir
if not os.path.isdir(save_path):
    os.mkdir(save_path)

image_list = os.listdir(img_path)
index = 1
count = len(image_list)
for image_name in image_list:
    image_path = os.path.join(img_path, image_name)
    image = Image.open(image_path)

    image = data_transforms(image)
    image = image.reshape(1, 3, 256, 128).cuda()

    with torch.no_grad():
        output = model(image)
        logits = F.softmax(output, 1)
        value, preds = torch.max(logits, 1)
        if preds == 0 and value < thre1:
            # preds = torch.tensor(1).cuda()
            preds = 1
        if preds == 1 and value < thre2:
            preds = 0
    # if preds == 1:
    #   result=1
    # else:
    #   result=0
    if save == 1:
        if preds == 0:
            src_path = img_path + image_name
            dst_path = save_path + '/' + '01'
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            # 获取已保存的图像数量
            image_count = len(os.listdir(dst_path))
            # 如果已达到最大存储量，删除最旧的图像
            old_index = 0
            if image_count >= 30000:
                oldest_image = os.listdir(dst_path)[old_index]
                old_index = old_index + 1
                if old_index == 3:
                    old_index = 0
                os.remove(os.path.join(dst_path,oldest_image))
            copyfile(src_path, dst_path + '/' + image_name)
    
        if preds == 1:
            src_path = img_path + image_name
            dst_path = save_path + '/' + '02'
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            # 获取已保存的图像数量
            image_count = len(os.listdir(dst_path))
            # 如果已达到最大存储量，删除最旧的图像
            old_index = 0
            if image_count >= 30000:
                oldest_image = os.listdir(dst_path)[old_index]
                old_index = old_index + 1
                if old_index == 3:
                    old_index = 0
                os.remove(os.path.join(dst_path,oldest_image))
            copyfile(src_path, dst_path + '/' + image_name)
    print(f"\r{index}/{count}", end="")
    index = index + 1

print("")




