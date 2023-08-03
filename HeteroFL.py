'''
Designed for (HeteroFL) / also to be (DepthFL)
No consideration for inconsistency
'''
import torch
# from torch import nn, autograd
# from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import torch
import copy

from torchvision.models import resnet18 as Presnet18
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet34 as Presnet34
from torchvision.models import ResNet34_Weights
from torchvision import datasets, transforms
import argparse
import os
import torchsummary
from math import sqrt
import wandb
from datetime import datetime

from models import *
from utils.fed import *
from utils.getData import *
from utils.util import test_img, extract_submodel_weight_from_globalH, get_logger
from utils.NeFedAvg import HeteroFL_Avg


parser = argparse.ArgumentParser()
parser.add_argument('--num_users', type=int, default=100)
parser.add_argument('--noniid', type=str, default='noniiddir') # noniid, noniiddir

parser.add_argument('--frac', type=float, default=0.1)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--local_bs', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0)
# parser.add_argument('--num_channels', type=int, default=1)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--mode', type=str, default='normal')

parser.add_argument('--rs', type=int, default=0)
parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")

parser.add_argument('--min_flex_num', type=int, default=2, help="0:0~ max(0,tc-args.min_flex_num)")
parser.add_argument('--max_flex_num', type=int, default=2, help="0:~4 min(tc+args.max_flex_num+1,5)")

parser.add_argument('--model_name', type=str, default='resnet18') # wide_resnet101_2
parser.add_argument('--device_id', type=str, default='3')

parser.add_argument('--local_ep', type=int, default=5)
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--wandb', type=bool, default=False)

parser.add_argument('--num_models', type=int, default=5)

parser.add_argument('--dataset', type=str, default='cifar10') # stl10, cifar10, svhn

args = parser.parse_args()
args.device = 'cuda:' + args.device_id

args.ps = [sqrt(0.2), sqrt(0.4), sqrt(0.6), sqrt(0.8), 1]
# args.ps = [0.2, 0.4, 0.6, 0.8, 1]

""" Vaying width of the network """
'''
network keys
- conv1.weight / bn1.weight/bias / bn1.running_mean / bn1.running_var / bn1.num_batches_tracked
- layerx.x. conv1.weight / bn1.weight/bias / bn1.running_mean / bn1.running_var / bn1.num_batches_tracked
            conv2.weight / bn2.weight/bias / bn2.running_mean / bn2.running_var / bn2.num_batches_tracked
- linear.weight/bias
shape = w[key].shape

len(shape) = 4: (conv1) / (layer.conv1 / layer.conv2)
len(shape) = 2: (linear.weight)
len(shape) = 1: bn1.weight/bias/running_mean/var [16/32/...] / (linear.bias) [10] / step_size
len(shape) = 0: bn1.num_batches_tracked
'''

dataset_train, dataset_test = getDataset(args)
if args.noniid == 'noniid':
    dict_users = cifar_noniid(args, dataset_train)
elif args.noniid == 'noniiddir':
    dict_users = cifar_noniiddir(args, 1, dataset_train)
else:
    dict_users = cifar_iid(dataset_train, args.num_users, args.rs)
# img_size = dataset_train[0][0].shape


def main():

    local_models = []
    if args.model_name == 'resnet18':
        for i in range(len(args.ps)):
            local_models.append(resnet18_HeteroFL(args.num_classes, args.ps[i]))
    elif args.model_name == 'resnet34':
        for i in range(len(args.ps)):
            local_models.append(resnet34_HeteroFL(args.num_classes, args.ps[i]))
    elif args.model_name == 'resnet56':
        for i in range(len(args.ps)):
            local_models.append(resnet56_HeteroFL(args.num_classes, args.ps[i]))
    elif args.model_name == 'resnet110':
        # args.epochs = 800
        for i in range(len(args.ps)):
            local_models.append(resnet110_HeteroFL(args.num_classes, args.ps[i]))
    elif args.model_name == 'wide_resnet101_2':
        for i in range(len(args.ps)):
            local_models.append(resnet101_2_HeteroFL(args.num_classes, args.ps[i]))
        
    '''    
    BN_layers = []
    Steps = []

    for i in range(len(local_models)):
        local_models[i].to(args.device)
        local_models[i].train()
        BN = {}
        Step = {}
        w = copy.deepcopy(local_models[i].state_dict())
        for key in w.keys():
            if len(w[key].shape)<=1 and key!='linear.bias' and not 'step' in key:
                BN[key] = w[key]
            elif 'step' in key:
                Step[key] = w[key]
        BN_layers.append(copy.deepcopy(BN))
        Steps.append(copy.deepcopy(Step))
    '''
    
    if args.model_name == 'resnet18':
        net_glob = resnet18_HeteroFL(args.num_classes, 1)
        if args.pretrained:
            w_glob = net_glob.state_dict()
            net_glob_temp = Presnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            net_glob_temp.fc = nn.Linear(512 * 1, 10)
            net_glob_temp.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)        
            w_glob_temp = net_glob_temp.state_dict()
            for key in w_glob.keys():
                w_glob[key] = w_glob_temp[key]
            net_glob.load_state_dict(w_glob)

    elif args.model_name == 'resnet34':
        net_glob = resnet34_HeteroFL(args.num_classes, 1)
        if args.pretrained:
            w_glob = net_glob.state_dict()
            net_glob_temp = Presnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            net_glob_temp.fc = nn.Linear(512 * 1, 10)
            net_glob_temp.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            w_glob_temp = net_glob_temp.state_dict()
            for key in w_glob.keys():
                w_glob[key] = w_glob_temp[key]
            net_glob.load_state_dict(w_glob)
        
    elif args.model_name == 'resnet56':
        net_glob = resnet56_HeteroFL(args.num_classes, 1)
    elif args.model_name == 'resnet110':
        net_glob = resnet110_HeteroFL(args.num_classes, 1)
    elif args.model_name == 'wide_resnet101_2':
        net_glob = resnet101_2_HeteroFL(args.num_classes, 1)

    net_glob.to(args.device)
    # torchsummary.summary(local_models[0], (3, 32, 32))
    net_glob.train()
    w_glob = net_glob.state_dict()
    
    # if args.pretrained:
    #     for i in range(len(local_models)):
    #         model_idx = i
    #         p_select = args.ps[model_idx]
    #         p_select_weight = extract_submodel_weight_from_globalH(net = copy.deepcopy(net_glob), p=p_select, model_i=model_idx)
    #         local_models[model_idx].load_state_dict(p_select_weight)    
    
    com_layers = []  # common layers: conv1, bn1, linear
    sing_layers = []  # singular layers: layer1.0.~ 
    bn_keys = []

    for i in w_glob.keys():
        if 'bn' not in i and 'shortcut.1' not in i:
            if 'layer' in i:
                sing_layers.append(i)
                '''
                layerX.Y.conv1.weight, layerX.Y.conv.weight
                layerX.0.shorcut.0.weight where X>=2
                '''
            else:
                com_layers.append(i)
                '''
                conv1.weight, fc.weight, fc.bias
                '''
        else:
            bn_keys.append(i)
            '''
            bn1.weight, bn1.bias
            layerX.Y.bn1.weight, layerX.Y.bn1.bias, layerX.Y.bn2.weight, layerX.Y.bn2.bias
            layerX.0.shortcut.1.weight, layerX.0.shortcut.1.bias where X>=2
            '''            
    loss_train = []

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.noniid == 'noniid': # noniid, noniiddir
        niid_name = '[niid]'
    elif args.noniid == 'noniiddir':
        niid_name = '[dir]'
    else:
        niid_name = '[iid]'
        
    if args.pretrained:
        args.model_name = 'P' + args.model_name
    args.name = '[' + str(args.dataset) + ']' + '[' + args.model_name + ']' + 'HeteroFL' + niid_name    
    filename = './output/heterofl/'+ timestamp + str(args.name) + str(args.rs)
    if not os.path.exists(filename):
        os.makedirs(filename)

    if args.wandb:
        run = wandb.init(dir=filename, project='HeteroFL-0803', name= str(args.name)+ str(args.rs), reinit=True, settings=wandb.Settings(code_dir="."))
        wandb.config.update(args)
    logger = get_logger(logpath=os.path.join(filename, 'logs'), filepath=os.path.abspath(__file__))

    lr = args.lr
    mlist = [_ for _ in range(args.num_models)]

    for iter in range(1,args.epochs+1):
        if iter == args.epochs/2:
            lr = lr*0.1
        elif iter == 3*args.epochs/4:
            lr = lr*0.1
        loss_locals = []

        # w_glob = net_glob.state_dict()
        w_locals = []
        w_locals.append([w_glob, args.num_models-1])
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # weight_for_bn = [] ####
        # step_weights = []
        
        for idx in idxs_users:
            if args.mode == 'worst':
                dev_spec_idx = 0
                model_idx = 0
            else:
                dev_spec_idx = idx//(args.num_users//args.num_models)
                model_idx = random.choice(mlist[max(0,dev_spec_idx-args.min_flex_num):min(len(args.ps),dev_spec_idx+1+args.max_flex_num)])
                # model_idx = random.choice(args.ps[max(0,dev_spec_idx-2):min(len(args.ps),dev_spec_idx+1+2)])
            p_select = args.ps[model_idx]
            
            p_select_weight = extract_submodel_weight_from_globalH(net = copy.deepcopy(net_glob), p=p_select, model_i=model_idx)
            # p_select_weight = p_submodel(net = copy.deepcopy(net_glob), BN_layer=BN_layers, p=p_select)
            model_select = local_models[model_idx]
            model_select.load_state_dict(p_select_weight)
            local = LocalUpdateH(args, dataset=dataset_train, idxs=dict_users[idx])
            weight, loss = local.train(net=copy.deepcopy(model_select).to(args.device), learning_rate=lr)
            # seperated_weights.append([copy.deepcopy(weight), model_idx])
            # weight_for_bn.append([model_idx, copy.deepcopy(weight)]) ####
            w_locals.append([copy.deepcopy(weight), model_idx])
            loss_locals.append(copy.deepcopy(loss))
        
        # a = copy.deepcopy(w_locals[0])
        # w_locals.append(a)
        # BN_layers = BN_update(weight_for_bn, BN_layers) ####
        # BN_layers, Steps = BN_update(seperated_weights, BN_layers, Steps, args)
        # w_glob = FedAvg(w_locals, BN_layers)
        w_glob = HeteroFL_Avg(w_locals, args, com_layers, sing_layers, bn_keys)

        net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)

        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))

        loss_train.append(loss_avg)
        #loss_train에 loss_avg 값 추가
        if iter % 10 == 0:
            if args.mode == 'worst': ##########
                ti = 1
            else: ##########
                ti = 5

            for ind in range(ti):
                p = args.ps[ind]
                model_e = copy.deepcopy(local_models[ind])
                
                f = extract_submodel_weight_from_globalH(net = copy.deepcopy(net_glob), p=p, model_i=ind)
                # f = p_submodel(net = copy.deepcopy(net_glob), BN_layer=BN_layers, p=p)
                model_e.load_state_dict(f)
                model_e.to(args.device)
                model_e.eval()
                acc_test, loss_test = test_img(model_e, dataset_test, args)
                print("Testing accuracy " + str(ind) + ": {:.2f}".format(acc_test))
                if args.wandb:
                    wandb.log({
                        "Communication round": iter,
                        "Local model " + str(ind) + " test accuracy": acc_test
                    })
    '''
    static BN (SBN)
    '''
    if args.mode == 'worst': ##########
        ti = 1
    else: ##########
        ti = 5

    for ind in range(ti):
        p = args.ps[ind]
        if args.model_name == 'resnet18':
            s2D = [[1, 1], [1, 1], [1, 1], [1, 1]]
            model_e = resnet18wd(s2D, p, False, args.num_classes)
        elif args.model_name == 'resnet34':
            s2D = [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1]]
            model_e = resnet34wd(s2D, p, False, args.num_classes)
        elif args.model_name == 'resnet56':
            s2D = [[1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)]
            model_e = resnet56wd(s2D, p, False, args.num_classes)
        elif args.model_name == 'resnet110':
            s2D = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)]
            model_e = resnet110wd(s2D, p, False, args.num_classes)
        elif args.model_name == 'wide_resnet101_2':
            s2D = [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1]]
            model_e = resnet101_2wd(s2D, p, False, args.num_classes)
        
        f = extract_submodel_weight_from_globalH(net = copy.deepcopy(net_glob), p=p, model_i=ind)
        model_e.load_state_dict(f, strict=False)
        model_e.to(args.device)
        model_e = sBN(model_e, dataset_train, args) # static batch normalization
        model_e.eval()
        acc_test, loss_test = test_img(model_e, dataset_test, args)
        print("Testing accuracy " + str(ind) + ": {:.2f}".format(acc_test))
        if args.wandb:
            wandb.log({
                "Communication round": iter+1,
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