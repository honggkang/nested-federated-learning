import math
# from math import ceil as up
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import copy
import logging


def up(value):
  return math.ceil(value)


# def test_img(net_g, datatest, args):
#     net_g.eval()
#     # testing
#     test_loss = 0
#     correct = 0

#     data_loader = DataLoader(datatest, batch_size=args.bs)
#     l = len(data_loader) # len(data_loader)= 469개, 128*468개 세트, 1개는 96개 들어있음
#     #print(l)
#     with torch.no_grad():
#       for idx, (data, target) in enumerate(data_loader):
#           if 'cuda' in args.device:
#               data, target = data.to(args.device), target.to(args.device)
#           logits, log_probs = net_g(data)
#           test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
#           y_pred = log_probs.data.max(1, keepdim=True)[1]

#           correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
#     test_loss /= len(data_loader.dataset)
#     accuracy = 100.00 * correct / len(data_loader.dataset)
#     return accuracy, test_loss


def test_img(net_g, datatest, args):
    net_g.eval()
    net_g.to(args.device)
    # testing
    test_loss = 0
    correct = 0

    data_loader = DataLoader(datatest, batch_size=args.bs)
    with torch.no_grad():
      for idx, (data, target) in enumerate(data_loader):
          if 'cuda' in args.device:
              data, target = data.to(args.device), target.to(args.device)
          logits, log_probs = net_g(data)
          test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
          y_pred = log_probs.data.max(1, keepdim=True)[1]

          correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    net_g.to('cpu')
    return accuracy, test_loss


def extract_submodel_weight_from_global(net, BN_layers, p, model_i):
  idx = model_i
  parent = net.state_dict()
  f = copy.deepcopy(parent)
  for key in parent.keys():
    shape = parent[key].shape
    if len(shape) == 4:
      if key == 'conv1.weight':
        f[key] = parent[key][0:up(shape[0]*p), :, :, :]
      else:
        f[key] = parent[key][0:up(shape[0]*p), 0:up(shape[1]*p), :, :]

    elif len(shape) == 2: # linear.weight len:2, shape [10, 64]
      f[key] = parent[key][:, 0:up(shape[1]*p)]

    elif len(shape) == 1:
        # bn1.weight/bias/running_mean/running_var, layer1.x len: 1 shape[0]: 16, layer2.0.bn2.bias len: 1 shape[0]: 32
        if key != 'fc.bias':
            f[key] = BN_layers[idx][key]
        # 'linear.bias' len 1 shape [10]
        else:
            f[key] = parent[key]  
        
    # bn1.num_batches_tracked len: 0 shape[0]: None
    else:
        f[key] = BN_layers[idx][key]
      
  return f


def extract_submodel_weight_from_globalM(net, BN_layer, Step_layer, p, model_i):
  idx = model_i
  parent = net.state_dict()
  f = copy.deepcopy(parent)
  for key in parent.keys():
    shape = parent[key].shape
    if len(shape) == 4:
      if key == 'conv1.weight':
        f[key] = parent[key][0:up(shape[0]*p), :, :, :]
      else:
        f[key] = parent[key][0:up(shape[0]*p), 0:up(shape[1]*p), :, :]

    elif len(shape) == 2: # linear.weight len:2, shape [10, 64]
      f[key] = parent[key][:, 0:up(shape[1]*p)]

    elif len(shape) == 1:
        # bn1.weight/bias/running_mean/running_var, layer1.x len: 1 shape[0]: 16, layer2.0.bn2.bias len: 1 shape[0]: 32
        if key != 'fc.bias' and not 'step' in key:
            f[key] = BN_layer[idx][key]
        # 'linear.bias' len 1 shape [10]
        elif 'step' in key:
            f[key] = Step_layer[idx][key]
        else:
            f[key] = parent[key]  
        
    # bn1.num_batches_tracked len: 0 shape[0]: None
    else:
        f[key] = BN_layer[idx][key]
      
  return f


def extract_submodel_weight_from_globalM2(net, target_net, BN_layer, Step_layer, model_i):
  idx = model_i
  parent = net.state_dict()
  f = target_net.state_dict()
  # f = copy.deepcopy(parent)
  for key in parent.keys():
    shape = f[key].shape
    if len(shape) == 4:
      if key == 'conv1.weight':
        f[key] = parent[key][0:shape[0], :, :, :]
      else:
        f[key] = parent[key][0:shape[0], 0:shape[1], :, :]

    elif len(shape) == 2: # linear.weight len:2, shape [10, 64]
      f[key] = parent[key][:, 0:shape[1]]

    elif len(shape) == 1:
        # bn1.weight/bias/running_mean/running_var, layer1.x len: 1 shape[0]: 16, layer2.0.bn2.bias len: 1 shape[0]: 32
        if key != 'fc.bias' and not 'step' in key:
            f[key] = BN_layer[idx][key]
        # 'linear.bias' len 1 shape [10]
        elif 'step' in key:
            f[key] = Step_layer[idx][key]
        else:
            f[key] = parent[key]  
        
    # bn1.num_batches_tracked len: 0 shape[0]: None
    else:
        f[key] = BN_layer[idx][key]
      
  return f


def extract_submodel_weight_from_globalM_vit(net, target_net, Norm_layers, Step_layer, p, model_i):
  idx = model_i
  parent = net.state_dict()
  f = target_net.state_dict()
  for key in parent.keys():
    shape = f[key].shape
    if len(shape) == 4: # patch_embedding.proj.weight
      oi = shape[0]
      f[key] = parent[key][0:oi, :, :, :]
    elif len(shape) == 3: # cls_token, pos_embed
      oi = shape[2]
      f[key] = parent[key][:,:,0:oi]
    elif len(shape) == 2: # attn.qkv.weight, attn.proj.weight, mlp.fc.weight
      ii = shape[1]
      oi = shape[0]
      if key == 'head.weight':
        f[key] = parent[key][:,0:ii]
      else:
        f[key] = parent[key][0:oi,0:ii]
    elif len(shape) == 1:
        if key == 'head.bias':
          f[key] = parent[key]
        elif 'norm' in key: # norm1.weight/bias, norm2.weight/bias
          f[key] = Norm_layers[idx][key]
        elif not 'step' in key: # patch_embedding.proj.bias, attn.qkv.bias, attn.proj.bias, mlp.fc.bias
          ii = shape[0]
          f[key] = parent[key][0:ii]
        elif 'step' in key:
            f[key] = Step_layer[idx][key]
        else:
            print('????', key, '????')
        
    # bn1.num_batches_tracked len: 0 shape[0]: None
    else:
        print('????', key, '????')
        f[key] = Norm_layers[idx][key]
      
  return f

####
def extract_submodel_weight_from_globalM_vitpy(net, target_net, Norm_layers, Step_layer, p, model_i):
  idx = model_i
  parent = net.state_dict()
  f = target_net.state_dict()
  for key in parent.keys():
    shape = f[key].shape
    if len(shape) == 4: # patch_embedding.proj.weight
      oi = shape[0]
      f[key] = parent[key][0:oi, :, :, :]
    elif len(shape) == 3: # cls_token, pos_embed
      oi = shape[2]
      f[key] = parent[key][:,:,0:oi]
    elif len(shape) == 2: # attn.qkv.weight, attn.proj.weight, mlp.fc.weight
      ii = shape[1]
      oi = shape[0]
      if key == 'heads.head.weight':
        f[key] = parent[key][:,0:ii]
      else:
        f[key] = parent[key][0:oi,0:ii]
    elif len(shape) == 1:
        '''
        ln, self_attention.in_proj_bias, self_attention.out_proj.bias, mlp.0.bias, mlp.3.bias, conv_proj.bias
        heads.head.bias
        '''      
        if key == 'heads.head.bias':
          f[key] = parent[key]
        elif 'ln' in key: # norm1.weight/bias, norm2.weight/bias
          f[key] = Norm_layers[idx][key]
        elif not 'step' in key: # self_attention.in/out_proj.bias, conv_proj.bias, mlp.0/3.bias
          ii = shape[0]
          f[key] = parent[key][0:ii]
        elif 'step' in key:
            f[key] = Step_layer[idx][key]
        else:
            print('????', key, '????')
        
    # bn1.num_batches_tracked len: 0 shape[0]: None
    else:
        print('????', key, '????')
        f[key] = Norm_layers[idx][key]
      
  return f


## HeteroFL

def extract_submodel_weight_from_globalH(net, p, model_i):
  idx = model_i
  parent = net.state_dict()
  f = copy.deepcopy(parent)
  for key in parent.keys():
    shape = parent[key].shape
    if len(shape) == 4:
      if key == 'conv1.weight':
        f[key] = parent[key][0:up(shape[0]*p), :, :, :]
      else:
        f[key] = parent[key][0:up(shape[0]*p), 0:up(shape[1]*p), :, :]

    elif len(shape) == 2: # linear.weight len:2, shape [10, 64]
      f[key] = parent[key][:, 0:up(shape[1]*p)]

    elif len(shape) == 1:
        # bn1.weight/bias/running_mean/running_var, layer1.x len: 1 shape[0]: 16, layer2.0.bn2.bias len: 1 shape[0]: 32
        if key != 'fc.bias':
            f[key] = parent[key][0:up(shape[0]*p)]
        # 'linear.bias' len 1 shape [10]
        else:
            f[key] = parent[key]
          
  return f


def p_submodel(net, BN_layer, p): # BN layer은 해당 p에 저장된 것을 가져옴
  index = int(5*p)-1
  a1 = net.state_dict()
  f = copy.deepcopy(a1)
  for key in a1.keys():
    shape = a1[key].shape
    if len(shape) == 4:
      if key == 'conv1.weight':
        f[key] = a1[key][0:up(shape[0]*p), :, :, :]
      else:
        f[key] = a1[key][0:up(shape[0]*p), 0:up(shape[1]*p), :, :]

    elif len(shape) ==2: # linear.weight
      f[key] = a1[key][:, 0:up(shape[1]*p)]
    else:
      if len(shape)>0 and (shape[0]>10): # bn1.weight, bias, running_mean, running_var, layer.1.0.bn1.weight, ....
        f[key] = a1[key][0:up(shape[0]*p)]
      else:  # bn1.num_batches_tracked
        f[key] = a1[key]
  for key in a1.keys():
    if len(a1[key].shape)<=1 and key!='linear.bias':
      f[key] = BN_layer[index][key]
  return f


def change(originw, w, w1, BN_layer, p_max, p_select):
  pmax_index = int(5*p_max)-1
  for key in w.keys():
    shape = originw[key].shape
    if len(shape) == 4:
      if key == 'conv1.weight':
        w[key][0:up(shape[0]*p_select), :, :, :] = w1[key]
      else:
        w[key][0:up(shape[0]*p_select), 0:up(shape[1]*p_select), :, :] = w1[key]
    elif len(shape) ==2:
      w[key][:, 0:up(shape[1]*p_select)] = w1[key]
    else:
      if len(shape)>0 and (shape[0]>10):
        w[key][0:up(shape[0]*p_select)] = w1[key]
      else:
        w[key] = w1[key]
  for key in w.keys():
    if len(w[key].shape)<=1 and key!='linear.bias':
      #if torch.allclose(w[key], BN_layer[4][key]):
        #print('same', key)
      w[key] = BN_layer[pmax_index][key]  
  return w


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        # info_file_handler.terminator = ""
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        # console_handler.terminator = ""
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger