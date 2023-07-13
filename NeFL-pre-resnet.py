'''
Difference w/ preMAFL-toy.py
- scheduler, optimizer loaded (warmup steps)
- noniid
- 224x224
- torch.nn.utils.clip_grad_norm(net.parameters(), 1)
0521: extract_submodel_weight_from_globalM2
'''
import numpy as np
import random
import torch
import copy

from torchvision import datasets, transforms
from torchvision.models import resnet18 as Presnet18
from torchvision.models import resnet34 as Presnet34
from torchvision.models import resnet101 as Presnet101
from torchvision.models import wide_resnet101_2 as Pwide_resnet101_2
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet101_Weights, Wide_ResNet101_2_Weights

import argparse
import os
import torchsummary
from math import sqrt
import wandb
from datetime import datetime

from models import *
from utils.fed import *
from utils.getData import *
from utils.util import test_img, extract_submodel_weight_from_globalM2, get_logger
from utils.NeFedAvg import NeFedAvg

parser = argparse.ArgumentParser()
parser.add_argument('--num_users', type=int, default=10) # 10 100
parser.add_argument('--noniid', action='store_true') # default: false
parser.add_argument('--class_per_each_client', type=int, default=10)

parser.add_argument('--frac', type=float, default=1) # 1 0.1
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--local_bs', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--epochs', type=int, default=100) # 100 500
parser.add_argument('--local_ep', type=int, default=1) # 1 5
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--lr', type=float, default=3e-2) # 1e-1 3e-2
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--mode', type=str, default='normal')
parser.add_argument('--rs', type=int, default=0)

parser.add_argument('--min_flex_num', type=int, default=0, help="0:0~ max(0,tc-args.min_flex_num)")
parser.add_argument('--max_flex_num', type=int, default=0, help="0:~4 min(tc+args.max_flex_num+1,5)")

parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")
parser.add_argument('--model_name', type=str, default='wide_resnet101_2') # wide_resnet101_2 resnet101 resnet18
parser.add_argument('--device_id', type=str, default='1')
parser.add_argument('--learnable_step', type=bool, default=True)
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--wandb', type=bool, default=False)

parser.add_argument('--dataset', type=str, default='cifar10') # stl10, cifar10, svhn
parser.add_argument('--method', type=str, default='DD') # DD, W, WD / fjord, depthfl

parser.add_argument('--name', type=str, default='[FjORD][cifar10][WR101]')
parser.add_argument('--warmup_steps', type=int, default=500)

    
args = parser.parse_args()
args.device = 'cuda:' + args.device_id
# args.ps = [sqrt(0.2), sqrt(0.4), sqrt(0.6), sqrt(0.8), 1] # only width -> param. size [0.2, 0.4, 0.6, 0.8, 1]
# args.ps = [0.2, 0.4, 0.6, 0.8, 1]
# parameter size gets 1/r^2 [r1, r2, r3, r4, r5]

""" Vaying width of the network """
'''
network keys
- conv1.weight / bn1.weight/bias / bn1.running_mean / bn1.running_var / bn1.num_batches_tracked
- layerx.x. conv1.weight / bn1.weight/bias / bn1.running_mean / bn1.running_var / bn1.num_batches_tracked
            conv2.weight / bn2.weight/bias / bn2.running_mean / bn2.running_var / bn2.num_batches_tracked
- linear.weight/bias => fc.weight/bias
shape = w[key].shape

len(shape) = 4: (conv1) / (layer.conv1 / layer.conv2)
len(shape) = 2: (linear.weight)
len(shape) = 1: bn1.weight/bias/running_mean/var [16/32/...] / (linear.bias) [10] / step_size
len(shape) = 0: bn1.num_batches_tracked
'''

if args.dataset =='cifar10':
    ## CIFAR
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset_train = datasets.CIFAR10('./.data/cifar', train=True, download=True, transform=transform_train)
    dataset_test = datasets.CIFAR10('./.data/cifar', train=False, download=True, transform=transform_test)

elif args.dataset == 'svhn':
    ### SVHN
    transform_train = transforms.Compose([
            transforms.Pad(padding=2),
            transforms.RandomCrop(size=(32, 32)),
            transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])
    dataset_train = datasets.SVHN('/home/hong/NeFL/.data/svhn', split='train', download=True, transform=transform_train)
    dataset_test = datasets.SVHN('/home/hong/NeFL/.data/svhn', split='test', download=True, transform=transform_test)

elif args.dataset == 'stl10':
    ### STL10
    transform_train = transforms.Compose([
                    transforms.RandomCrop(96, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465],
                                        [0.2471, 0.2435, 0.2616])
                ])
    transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465],
                                        [0.2471, 0.2435, 0.2616])
            ])
    dataset_train = datasets.STL10('/home/hong/NeFL/.data/stl10', split='train', download=True, transform=transform_train)
    dataset_test = datasets.STL10('/home/hong/NeFL/.data/stl10', split='test', download=True, transform=transform_test)

if args.noniid:
    dict_users = cifar_noniid(args, dataset_train)
else:
    dict_users = cifar_iid(dataset_train, args.num_users, args.rs)
# img_size = dataset_train[0][0].shape


def main():
    if args.model_name == 'wide_resnet101_2':
        if args.method == 'DD':
            args.ps = [1, 1, 1]
            args.s2D = [ # 101_2 / 50, 75, 100
                [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0]] ], # 50.818%
                [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1]] ], # 75.406%
                [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1]] ]
            ]
        elif args.method == 'W':
            args.ps = [sqrt(0.5), sqrt(0.75), 1]
            args.s2D = [ # 101_2 / 50, 75, 100
                [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1]] ],
                [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1]] ],
                [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1]] ]
            ]
    
    
    args.num_models = len(args.ps)

    local_models = []
    if args.model_name == 'resnet18':
        for i in range(len(args.ps)):
            local_models.append(resnet18wd(args.s2D[i][0], args.ps[i], args.learnable_step, args.num_classes))
    elif args.model_name == 'resnet34':
        for i in range(len(args.ps)):
            local_models.append(resnet34wd(args.s2D[i][0], args.ps[i], args.learnable_step, args.num_classes))
    elif args.model_name == 'resnet101':
        for i in range(len(args.ps)):
            local_models.append(resnet101wd(args.s2D[i][0], args.ps[i], args.learnable_step, args.num_classes))
    elif args.model_name == 'wide_resnet101_2':
        for i in range(len(args.ps)):
            local_models.append(resnet101_2wd(args.s2D[i][0], args.ps[i], args.learnable_step, args.num_classes))


    BN_layers = []
    Steps = []

    for i in range(len(local_models)):
        local_models[i].train()
        BN = {}
        Step = {}
        w = copy.deepcopy(local_models[i].state_dict())
        for key in w.keys():
            if len(w[key].shape)<=1 and key!='fc.bias' and not 'step' in key:
                BN[key] = w[key]
            elif 'step' in key:
                Step[key] = w[key]
        BN_layers.append(copy.deepcopy(BN))
        Steps.append(copy.deepcopy(Step))
        local_models[i].to(args.device)

    if args.model_name == 'resnet18':
        net_glob = resnet18wd(args.s2D[-1][0], 1, True, num_classes=args.num_classes)        
        w_glob = net_glob.state_dict()
        if args.pretrained:
            net_glob_temp = Presnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            net_glob_temp = Presnet18(weights=None)
        net_glob_temp.fc = nn.Linear(512 * 1, args.num_classes)
        net_glob_temp.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        w_glob_temp = net_glob_temp.state_dict()
        for key in w_glob_temp.keys():
            w_glob[key] = w_glob_temp[key]
        net_glob.load_state_dict(w_glob)
    elif args.model_name== 'resnet34':
        net_glob = resnet34wd(args.s2D[-1][0], 1, True, num_classes=args.num_classes)
        w_glob = net_glob.state_dict()
        if args.pretrained:
            net_glob_temp = Presnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            net_glob_temp = Presnet34(weights=None)
        net_glob_temp.fc = nn.Linear(512 * 1, args.num_classes)
        net_glob_temp.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        w_glob_temp = net_glob_temp.state_dict()
        for key in w_glob_temp.keys():
            w_glob[key] = w_glob_temp[key]
        net_glob.load_state_dict(w_glob)
    elif args.model_name== 'resnet101':
        net_glob = resnet101wd(args.s2D[-1][0], 1, True, num_classes=args.num_classes)
        w_glob = net_glob.state_dict()
        if args.pretrained:
            net_glob_temp = Presnet101(weights=ResNet101_Weights.IMAGENET1K_V2) ########
        else:
            net_glob_temp = Presnet101(weights=None)
        net_glob_temp.fc = nn.Linear(512 * 4, args.num_classes)
        w_glob_temp = net_glob_temp.state_dict()
        for key in w_glob_temp.keys():
            w_glob[key] = w_glob_temp[key]
        net_glob.load_state_dict(w_glob)
    elif args.model_name == 'wide_resnet101_2':
        net_glob = resnet101_2wd(args.s2D[-1][0], 1, True, num_classes=args.num_classes)
        w_glob = net_glob.state_dict()
        if args.pretrained:
            net_glob_temp = Pwide_resnet101_2(weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1)
        else:
            net_glob_temp = Pwide_resnet101_2(weights=None)

        net_glob_temp.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net_glob_temp.fc = nn.Linear(2048, args.num_classes)
        w_glob_temp = net_glob_temp.state_dict()
        for key in w_glob_temp.keys():
            w_glob[key] = w_glob_temp[key]
        net_glob.load_state_dict(w_glob)
    # torchsummary.summary(net_glob_temp, (3, 32, 32), device='cpu')

    if args.pretrained:
        for i in range(len(local_models)):
            model_idx = i
            p_select = args.ps[model_idx]
            p_select_weight = extract_submodel_weight_from_globalM2(net = copy.deepcopy(net_glob), target_net=copy.deepcopy(local_models[model_idx]), BN_layer=BN_layers, Step_layer=Steps, model_i=model_idx)
            # p_select_weight = extract_submodel_weight_from_global(net = copy.deepcopy(net_glob), BN_layer=BN_layers, p=p_select, model_i=model_idx)
            local_models[model_idx].load_state_dict(p_select_weight)            

    # net_glob.to(args.device)
    # torchsummary.summary(local_models[0], (3, 32, 32)) # device='cpu'
    net_glob.train()

    w_glob = net_glob.state_dict()
    
    com_layers = []  # common layers: conv1, bn1, linear
    sing_layers = []  # singular layers: layer1.0.~ 
    bn_keys = []
    step_keys = []
            
    for i in w_glob.keys():
        if 'bn' not in i and 'downsample.1' not in i and 'step' not in i:
            if 'layer' in i:
                sing_layers.append(i)
            else:
                com_layers.append(i)
        elif 'step' in i:
            step_keys.append(i)
        else:
            bn_keys.append(i)
            
    loss_train = []

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = './output/mafl/'+ timestamp + str(args.name) + str(args.rs)
    if not os.path.exists(filename):
        os.makedirs(filename)

    if args.wandb:
        run = wandb.init(dir=filename, project='PreNeFL-0712', name= str(args.name)+ str(args.rs), reinit=True, settings=wandb.Settings(code_dir="."))
        wandb.config.update(args)
    logger = get_logger(logpath=os.path.join(filename, 'logs'), filepath=os.path.abspath(__file__))

    lr = args.lr
    mlist = [_ for _ in range(args.num_models)]
    scheduler_all = {}
    optimizer_all = {}

    for iter in range(1,args.epochs+1):
        loss_locals = []
        w_locals = []
        w_locals.append([w_glob, args.num_models-1])
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            if args.mode == 'worst':
                dev_spec_idx = 0
                model_idx = 0
            else:
                dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
                model_idx = random.choice(mlist[max(0,dev_spec_idx-args.min_flex_num):min(len(args.ps),dev_spec_idx+1+args.max_flex_num)])
            p_select = args.ps[model_idx]
            model_select = local_models[model_idx]
            p_select_weight = extract_submodel_weight_from_globalM2(net = copy.deepcopy(net_glob), target_net=copy.deepcopy(model_select), BN_layer=BN_layers, Step_layer=Steps, model_i=model_idx)

            model_select.load_state_dict(p_select_weight)
            local = LocalUpdateM_ft(args, dataset=dataset_train, idxs=dict_users[idx])
            if iter==1:
                weight, loss, new_scheduler_state, new_optimizer_state = local.train(net=copy.deepcopy(model_select), learning_rate=lr)
            else:
                weight, loss, new_scheduler_state, new_optimizer_state = local.train(net=copy.deepcopy(model_select), learning_rate=lr, load_scheduler=scheduler_all[idx], load_optimizer=optimizer_all[idx])
            
            w_locals.append([copy.deepcopy(weight), model_idx])
            print('IDX: {}, MODEL: {}, LOSS: {:.4f}'.format(idx, model_idx, loss)) # print(idx, model_idx, loss)
            loss_locals.append(copy.deepcopy(loss))
            scheduler_all[idx] = copy.deepcopy(new_scheduler_state)
            optimizer_all[idx] = copy.deepcopy(new_optimizer_state)
        w_glob, BN_layers, Steps = NeFedAvg(w_locals, BN_layers, Steps, args, com_layers, sing_layers)

        net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)

        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        if iter % 4 == 0:
            if args.mode == 'worst': ##########
                ti = 1
            else: ##########
                ti = args.num_models

            for ind in range(ti):
                p = args.ps[ind]
                model_e = copy.deepcopy(local_models[ind])               
                f = extract_submodel_weight_from_globalM2(net = copy.deepcopy(net_glob), target_net=copy.deepcopy(model_e), BN_layer=BN_layers, Step_layer=Steps, model_i=ind)
                model_e.load_state_dict(f)
                model_e.eval()
                acc_test, loss_test = test_img(model_e, dataset_test, args)
                print("Testing accuracy " + str(ind) + ": {:.2f}".format(acc_test))
                if args.wandb:
                    wandb.log({
                        "Communication round": iter,
                        "Local model " + str(ind) + " test accuracy": acc_test
                    })

    # testing
    net_glob.eval()

    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
  
    if args.wandb:
        run.finish()


if __name__ == "__main__":
    for i in range(args.num_experiment):
        torch.manual_seed(args.rs)
        torch.cuda.manual_seed(args.rs)
        torch.cuda.manual_seed_all(args.rs) # if use multi-GPU
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        np.random.seed(args.rs)
        random.seed(args.rs)
        main()
        args.rs = args.rs+1