
from torchvision import datasets, transforms
from models.pyvit import my_vit_b_16
from torchvision.models import vit_b_16, ViT_B_16_Weights

from collections import OrderedDict
import torch.nn as nn
import torch
import numpy as np
import random
import copy
import argparse
import os
# import torchsummary
# from torchsummaryX import summary
from math import sqrt
import wandb
from datetime import datetime

from models import *
from utils.fed import *
from utils.getData import *
from utils.util import test_img, extract_submodel_weight_from_globalM_vitpy, get_logger
from utils.NeFedAvg import NeFedAvg_vit


parser = argparse.ArgumentParser()
parser.add_argument('--num_users', type=int, default=10)
parser.add_argument('--noniid', type=str, default='noniiddir') # noniid, noniiddir
parser.add_argument('--class_per_each_client', type=int, default=8) # for noniid dataset
parser.add_argument('--frac', type=float, default=1)

parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--local_bs', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--local_ep', type=int, default=1)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--lr', type=float, default=3e-2)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--mode', type=str, default='normal')
parser.add_argument('--rs', type=int, default=0)

parser.add_argument('--min_flex_num', type=int, default=0, help="0:0~ max(0,tc-args.min_flex_num)")
parser.add_argument('--max_flex_num', type=int, default=0, help="0:~4 min(tc+args.max_flex_num+1,5)")

parser.add_argument('--num_experiment', type=int, default=2, help="the number of experiments")
parser.add_argument('--model_name', type=str, default='vit_b_16')
parser.add_argument('--device_id', type=str, default='3')
parser.add_argument('--learnable_step', type=bool, default=False)
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--wandb', type=bool, default=True)

parser.add_argument('--dataset', type=str, default='cifar10') # stl10, cifar10, svhn
parser.add_argument('--method', type=str, default='DD') # DD, W, WD / fjord, depthfl

# parser.add_argument('--name', type=str, default='[FjORD][vit_b16][iid]') # 
# parser.add_argument('--num_models', type=int, default=1)

parser.add_argument('--warmup_steps', type=int, default=500)

args = parser.parse_args()
args.device = 'cuda:' + args.device_id

""" Vaying width/depth of the network """
'''
network keys
- conv_proj.weight
- class_token, encoder.pos_embedding
- self_attention.in_proj_weight, self_attention.out_proj.weight, mlp.0.weight, mlp.3.weight
- heads.head.weight
- ln, self_attention.in_proj_bias, self_attention.out_proj.bias, mlp.0.bias, mlp.3.bias, conv_proj.bias
- heads.head.bias

shape = w[key].shape

len(shape) = 4: conv_proj.weight
len(shape) = 2: (linear.weight)
len(shape) = 1: bn1.weight/bias/running_mean/var [16/32/...] / (linear.bias) [10] / step_size
len(shape) = 0: bn1.num_batches_tracked
'''

# transforms.RandomHorizontalFlip(),
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

if args.noniid == 'noniid':
    dict_users = cifar_noniid(args, dataset_train)
elif args.noniid == 'noniiddir':
    dict_users = cifar_noniiddir(args, 0.5, dataset_train)
else:
    dict_users = cifar_iid(dataset_train, args.num_users, args.rs)


def main():
    if args.method == 'W':
        args.ps = [sqrt(0.5), sqrt(0.75), 1]
        args.s2D = [
                [1,1,1,1,1,1,1,1,1,1,1,1] ,
                [1,1,1,1,1,1,1,1,1,1,1,1] ,
                [1,1,1,1,1,1,1,1,1,1,1,1]
                ]
    elif args.method == 'DD':
        args.ps = [1, 1, 1]
        args.s2D = [ # 50 75 100
                [1,1,1,1,1,1,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0],
                [1,1,1,1,1,1,1,1,1,1,1,1]
            ]
    elif args.method == 'WD':
        args.ps = [sqrt(0.747389751896896), sqrt(0.898744459091891), 1]
        args.s2D = [
                [1,1,1,1,1,1,1,1,0,0,0,0] ,
                [1,1,1,1,1,1,1,1,1,1,0,0] ,
                [1,1,1,1,1,1,1,1,1,1,1,1]
            ]
    args.num_models = len(args.ps)
    # torchsummary.summary(test_net, (3,224,224))
    # summary(test_net, torch.zeros(1,3,224,224))
    local_models = []
    if args.model_name == 'vit_b_16':
        for i in range(len(args.ps)):
            local_models.append(my_vit_b_16(step_size_vector=args.s2D[i], width_ratio=args.ps[i], learnable_step=args.learnable_step))
    
    Norm_layers = []
    Steps = []

    for i in range(len(local_models)):
        # local_models[i].to(args.device)
        local_models[i].train()
        Norm = {}
        Step = {}
        w = copy.deepcopy(local_models[i].state_dict())
        for key in w.keys():
            if 'ln' in key: # len(w[key].shape)<=1 and 
                Norm[key] = w[key]
            elif 'step' in key:
                Step[key] = w[key]
        Norm_layers.append(copy.deepcopy(Norm))
        Steps.append(copy.deepcopy(Step))

    net_glob = copy.deepcopy(local_models[-1])
    
    net_glob_temp = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    # net_glob_temp.to(args.device)
    hidden_dim = 768 # vit_b_16
    heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
    heads_layers["head"] = nn.Linear(hidden_dim, args.num_classes)
    net_glob_temp.heads = nn.Sequential(heads_layers)
    
    w_glob_temp = net_glob_temp.state_dict()

    if args.pretrained:
        w = net_glob.state_dict()
        for key in w_glob_temp.keys():
            shape = w[key].shape
            if len(shape) == 4:
                '''
                conv_proj.weight
                '''
                oi = shape[0]
                w[key] = w_glob_temp[key][0:oi,:,:,:] # only output
            elif len(shape) == 3:
                '''
                class_token, encoder.pos_embedding
                '''
                oi = shape[2]
                w[key] = w_glob_temp[key][:,:,0:oi]
            elif len(shape) == 2:
                '''
                self_attention.in_proj_weight, self_attention.out_proj.weight, mlp.0.weight, mlp.3.weight
                heads.head.weight
                '''
                ii = shape[1]
                oi = shape[0]
                w[key] = w_glob_temp[key][0:oi,0:ii]
            elif len(shape) == 1:
                '''
                ln, self_attention.in_proj_bias, self_attention.out_proj.bias, mlp.0.bias, mlp.3.bias, conv_proj.bias
                heads.head.bias
                '''
                ii = shape[0]
                w[key] = w_glob_temp[key][0:ii]
        net_glob.load_state_dict(w)                    
        
        for i in range(len(local_models)):
            w = copy.deepcopy(local_models[i].state_dict())
            for key in w_glob_temp.keys():
                shape = w[key].shape
                # print(key, len(shape))
                if len(shape) == 4:
                    '''
                    conv_proj.weight
                    '''
                    oi = shape[0]
                    w[key] = w_glob_temp[key][0:oi,:,:,:]
                elif len(shape) == 3:
                    '''
                    class_token, encoder.pos_embedding
                    '''
                    oi = shape[2]
                    w[key] = w_glob_temp[key][:,:,0:oi]
                elif len(shape) == 2:
                    '''
                    self_attention.in_proj_weight, self_attention.out_proj.weight, mlp.0.weight, mlp.3.weight
                    heads.head.weight
                    '''
                    ii = shape[1]
                    oi = shape[0]
                    if key == 'heads.head.weight':
                        w[key] = w_glob_temp[key][:,0:ii]
                    else:
                        w[key] = w_glob_temp[key][0:oi,0:ii]
                elif len(shape) == 1:
                    '''
                    ln, self_attention.in_proj_bias, self_attention.out_proj.bias, mlp.0.bias, mlp.3.bias, conv_proj.bias
                    heads.head.bias
                    '''
                    if key == 'heads.head.bias':
                        w[key] = w_glob_temp[key]
                    elif 'ln' in key:
                        ii = shape[0]
                        Norm_layers[i][key]=w_glob_temp[key][0:ii]
                        w[key] = w_glob_temp[key][0:ii]
                    else:
                        ii = shape[0]
                        w[key] = w_glob_temp[key][0:ii]
            local_models[i].load_state_dict(w)
            # input = torch.randn(1, 3, 224, 224).to(args.device)
        # input = torch.ones(1, 3, 224, 224).to(args.device)
        # test_out = local_models[0](input)
        # print(test_out)
            
    # net_glob.to(args.device)
    net_glob.train()
    w_glob = net_glob.state_dict()
    # torchsummary.summary(local_models[0], (3, 32, 32)) # device='cpu'
    
    com_layers = []  # common-depth layers: cls_token, pos_embed, patch_embed, ...
    sing_layers = [] # singular-depth layers: blocks.1.0.~ 
    norm_keys = []
    step_keys = []
            
    for i in w_glob.keys():
        if 'ln' not in i and 'step' not in i:
            if 'encoder_layer' in i:
                sing_layers.append(i)
            else:
                com_layers.append(i)
        elif 'step' in i:
            step_keys.append(i)
        else:
            norm_keys.append(i)
            
    loss_train = []
    if args.method == 'W':
        if args.learnable_step:
            method_name = 'NeFLW'
        else:
            method_name = 'FjORD'
    elif args.method == 'DD':
        if args.learnable_step:
            method_name = 'NeFLADD'
        else:
            method_name = 'NeFLDD'
    elif args.method == 'OD':
        if args.learnable_step:
            method_name = 'NeFLAOD'
        else:
            method_name = 'NeFLOD'
    elif args.method == 'WD':
        if args.learnable_step:
            method_name = 'NeFLWD'
        else:
            method_name = 'NeFWDnL'
    
    if args.noniid == 'noniid': # noniid, noniiddir
        niid_name = '[niid]'
    elif args.noniid == 'noniiddir':
        niid_name = '[dir]'
    else:
        niid_name = '[iid]'    

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.name = '[' + str(args.dataset) + ']' + '[ViT_B16]' + method_name + niid_name

    filename = './output/neflvit/'+ timestamp + str(args.name) + str(args.rs)
    if not os.path.exists(filename):
        os.makedirs(filename)

    if args.wandb:
        run = wandb.init(dir=filename, project='NeFL_ViT0804', name= str(args.name)+ str(args.rs), reinit=True, settings=wandb.Settings(code_dir="."))
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
        
        for j, idx in enumerate(idxs_users):
            if args.mode == 'worst':
                dev_spec_idx = 0
                model_idx = 0
            else:
                dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
                model_idx = random.choice(mlist[max(0,dev_spec_idx-args.min_flex_num):min(len(args.ps),dev_spec_idx+1+args.max_flex_num)])
            p_select = args.ps[model_idx]
            model_select = local_models[model_idx]
            p_select_weight = extract_submodel_weight_from_globalM_vitpy(net = copy.deepcopy(net_glob), target_net = copy.deepcopy(model_select), Norm_layers=copy.deepcopy(Norm_layers), Step_layer=copy.deepcopy(Steps), p=p_select, model_i=model_idx)
            
            model_select.load_state_dict(p_select_weight)
            local = LocalUpdateM_pyvit(args, dataset=dataset_train, idxs=dict_users[idx])
            if iter==1:
                weight, loss, new_scheduler_state, new_optimizer_state = local.train(net=copy.deepcopy(model_select), round=iter, learning_rate=lr)
            else:
                weight, loss, new_scheduler_state, new_optimizer_state = local.train(net=copy.deepcopy(model_select), round=iter, learning_rate=lr, load_scheduler=scheduler_all[idx], load_optimizer=optimizer_all[idx])
            
            w_locals.append([copy.deepcopy(weight), model_idx])
            print('IDX: {}, MODEL: {}, LOSS: {:.4f}'.format(idx, model_idx, loss)) # print(idx, model_idx, loss)
            loss_locals.append(copy.deepcopy(loss))
            scheduler_all[idx] = copy.deepcopy(new_scheduler_state)
            optimizer_all[idx] = copy.deepcopy(new_optimizer_state)
        w_glob, Norm_layers, Steps = NeFedAvg_vit(w_locals, Norm_layers, Steps, args, com_layers, sing_layers, local_models)

        net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)

        print('Round {:3d}, Average loss {:.3f}, LR {}'.format(iter, loss_avg, new_scheduler_state['_last_lr'][0]))
        loss_train.append(loss_avg)
        if iter % 4 == 0:
            if args.mode == 'worst':
                ti = 1
            else:
                ti = args.num_models

            for ind in range(ti):
                p = args.ps[ind]
                model_e = copy.deepcopy(local_models[ind])
                f = extract_submodel_weight_from_globalM_vitpy(net = copy.deepcopy(net_glob), target_net = model_e, Norm_layers=copy.deepcopy(Norm_layers), Step_layer=copy.deepcopy(Steps), p=p_select, model_i=ind)
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