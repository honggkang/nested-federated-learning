#%%
import torch
from torch import nn
# from torch import nn, autograd
# from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
from models import *
import numpy as np

from utils.getData import *
import argparse
import os
import torchsummary
from math import sqrt
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--num_users', type=int, default=100)
parser.add_argument('--noniid', action='store_true') # default: false
parser.add_argument('--class_per_each_client', type=int, default=10)

# parser.add_argument('--frac', type=float, default=0.1)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--local_bs', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--local_ep', type=int, default=5)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--mode', type=str, default='normal')
parser.add_argument('--rs', type=int, default=0)

parser.add_argument('--min_flex_num', type=int, default=2, help="0:0~ max(0,tc-args.min_flex_num)")
parser.add_argument('--max_flex_num', type=int, default=2, help="0:~4 min(tc+args.max_flex_num+1,5)")

parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")
parser.add_argument('--model_name', type=str, default='resnet18') # wide_resnet101_2
parser.add_argument('--device_id', type=str, default='0')
parser.add_argument('--learnable_step', type=bool, default=True)
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--wandb', type=bool, default=True)
parser.add_argument('--model_num', type=int, default=5)
parser.add_argument('--method', type=str, default='DD')

    
# args = parser.parse_args()
args = parser.parse_args(args=[])
args.device = 'cuda:' + args.device_id

X = [[], [], [], [], []]
Xk = []
BNs = [[], [], [], [], []]
Steps = [[], [], [], [], []]

args.ps, args.s2D = get_submodel_info(args)

activation = {}
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# h1 = tmp_model.avgpool.register_forward_hook(getActivation('avgpool'))
# h2 = tmp_model.maxpool.register_forward_hook(getActivation('maxpool'))
# h3 = tmp_model.layer3[0].downsample[1].register_forward_hook(getActivation('comp'))

torch.manual_seed(args.rs)
torch.cuda.manual_seed(args.rs)
torch.cuda.manual_seed_all(args.rs) # if use multi-GPU
np.random.seed(args.rs)
random.seed(args.rs)

# filename = './output/nefl/' + '20230814-091553[cifar10][resnet18]NeFLADD[iid]0' + '/models'
filename = './output/nefl/' + '20230814-091622[cifar10][Presnet18]NeFLADD[iid]0' + '/models'        

if args.model_name == 'resnet18':
    for i in range(len(args.ps)):
        activation = {}
        # tmp_model = resnet18wd(args.s2D[-1][0], args.ps[i], True, num_classes=args.num_classes)
        tmp_model = resnet18wd(args.s2D[i][0], args.ps[i], True, num_classes=args.num_classes)
        tmp_model.load_state_dict(torch.load(os.path.join(filename, 'model' + str(i) + '.pt')))
        tmp_model.eval()
        
        tmp_model.layer1[0].register_forward_hook(getActivation('block1'))
        tmp_model.layer1[1].register_forward_hook(getActivation('block2'))
        tmp_model.layer2[0].register_forward_hook(getActivation('block3'))
        tmp_model.layer2[1].register_forward_hook(getActivation('block4'))
        tmp_model.layer3[0].register_forward_hook(getActivation('block5'))
        tmp_model.layer3[1].register_forward_hook(getActivation('block6'))
        tmp_model.layer4[0].register_forward_hook(getActivation('block7'))
        tmp_model.layer4[1].register_forward_hook(getActivation('block8'))
        
        out = tmp_model(torch.randn(1,3,32,32))

        list_activation = list(activation.values())
        for j in range(len(activation)):
            X[i].append(torch.norm(list_activation[j], p=2))
    
    
t = np.arange(1,len(X[0])+1,1)

plt.xlabel('Blocks')
plt.ylabel('L2 norm')
plt.plot(t, X[0], 'r--', label = 'Submodel 1')
plt.plot(t, X[1], 'g-.', label = 'Submodel 2')
plt.plot(t, X[2], 'bs:', label = 'Submodel 3')
plt.plot(t, X[3], 'y^-', label = 'Submodel 4')
plt.plot(t, X[4], 'k', label = 'Submodel 5')
plt.legend(loc='upper left')
plt.show()
print('Models Loaded')
# %%
