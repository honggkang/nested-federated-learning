import copy
import torch
# from math import ceil as up
from utils.util import up


def BN_update(weight_for_bn, BN_set):
    if len(weight_for_bn)!=10:
        print('BN update Error!, len is', len(weight_for_bn))
    a = [0]*len(weight_for_bn)
    num = [0]*len(BN_set)
    for j in range(10):
        select_idx = weight_for_bn[j][0]
        a[j] = select_idx
        num[select_idx] += 1
    # a_02 = a.count(0.2)
    # a_04 = a.count(0.4)
    # a_06 = a.count(0.6)
    # a_08 = a.count(0.8)
    # a_10 = a.count(1)

    # num = [a_02, a_04, a_06, a_08, a_10] # p_select 확률 별 개수
    # print('# of selected submodels: ', num)
    
    for k in range(len(BN_set)):
        if num[k] != 0:
            for key in BN_set[k].keys():
                BN_set[k][key] = 0*BN_set[k][key]

    for i in range(10):
        # index = int(5*weight_for_bn[i][0])-1 # p_select 0.2->0, 0.4->1, ..., 1->4
        index = weight_for_bn[i][0]
        for key in BN_set[index].keys():
            BN_set[index][key] = BN_set[index][key] + weight_for_bn[i][1][key]/num[index]
    
    return BN_set


def FedAvg1(w):
    print('lenw-1', len(w)-1)
    # w= w_locals를 받음
    w_avg = copy.deepcopy(w[0][0])
    for k in w_avg.keys():
        for i in range(1, len(w)-1):   # len(w_locals) :100
            w_avg[k] += w[i][0][k]
        w_avg[k] = torch.div(w_avg[k], len(w)-1)
    return w_avg


def FedAvg(w, BN_layer): # [0]: weight, [1]: dropout probability

  w_avg = copy.deepcopy(w[0][0])
  
  #for k in w_avg.keys():
  #  w_avg[k] = w_avg[k] - w_avg[k] # 전체 model parameter 초기화
  a = [0]*(len(w)-1)
  num = [len(w)-1]*5
  for n in range(1, len(w)):
    # a[n-1] = int(5*w[n][1])
    a[n-1] = w[n][1]
  b = [0]*5
  b[len(b)-1]=a.count(len(b)-1)
  for i in range(len(b)-1):
    b[i] = a.count(i)
    num[i+1] = num[i] - b[i]

  
  # [a1, ..., a5] = [2, 1, 2, 2, 3]. num = [10, 8, 7, 5, 3]
  zeroindex = 0
  for o in range(5):
    if num[o]==0:
      break
    zeroindex += 1
  prob = 0.2*(zeroindex)  # define a max size of model that is updated
  print("a:", a, "b:", b, "num:", num, 'maxP:', prob)
  # print('=================num, 0index, make0_submodel_size=================', num, zeroindex, prob)
  for key in w_avg.keys():
    '''
    Initialization
    w_glob에 대해, client가 업데이트한것에 대해선 0으로 초기화, 어떤 client도 건드리지 않은 파라미터는 그대로 가져감
    '''
    shape = w_avg[key].shape
    if len(shape) == 4:
      if key == 'conv1.weight':
        w_avg[key][0:up(shape[0]*prob), :, :, :] = w_avg[key][0:up(shape[0]*prob), :, :, :]-w_avg[key][0:up(shape[0]*prob), :, :, :]
      else: # other conv layers (layer1.0.conv1.weight -> shape = [16, 16, 3, 3]. layer2.0.conv1.weight -> hsape = [32, 16, 3, 3])
        w_avg[key][0:up(shape[0]*prob), 0:up(shape[1]*prob), :, :] = w_avg[key][0:up(shape[0]*prob), 0:up(shape[1]*prob), :, :]-w_avg[key][0:up(shape[0]*prob), 0:up(shape[1]*prob), :, :]

    elif len(shape) == 2: # linear.weight, 
      w_avg[key][:, 0:up(shape[1]*prob)] = w_avg[key][:, 0:up(shape[1]*prob)]-w_avg[key][:, 0:up(shape[1]*prob)]
    else: # bn1.weight
      if len(shape)>0 and (shape[0]>10):
        w_avg[key][0:up(shape[0]*prob)] = w_avg[key][0:up(shape[0]*prob)] - w_avg[key][0:up(shape[0]*prob)]
      else:
        w_avg[key] = w_avg[key] - w_avg[key]


  for key in w_avg.keys():
    shape = w_avg[key].shape
    for i in range(1, len(w)):
    #   p_index = int((5*w[i][1])-1)
      p_index = w[i][1]
      for j in range(p_index+1):
        if len(shape) == 4:
          if key == 'conv1.weight':
            #print('key found------------------------------------', key)
            w_avg[key][up(shape[0]*0.2*j):up(shape[0]*(0.2*j+0.2)), :, :, :] += w[i][0][key][up(shape[0]*0.2*j):up(shape[0]*(0.2*j+0.2)), :, :, :]/num[j]
          else:
            w_avg[key][up(shape[0]*0.2*j):up(shape[0]*(0.2*j+0.2)), up(shape[1]*0.2*j):up(shape[1]*(0.2*j+0.2)), :, :] += w[i][0][key][up(shape[0]*0.2*j):up(shape[0]*(0.2*j+0.2)), up(shape[1]*0.2*j):up(shape[1]*(0.2*j+0.2)), :, :]/num[j]
            if j!=0:
              w_avg[key][0:up(shape[0]*0.2*j), up(shape[1]*0.2*j):up(shape[1]*(0.2*j+0.2)), :, :] += w[i][0][key][0:up(shape[0]*0.2*j), up(shape[1]*0.2*j):up(shape[1]*(0.2*j+0.2)), :, :]/num[j]
              w_avg[key][up(shape[0]*0.2*j):up(shape[0]*(0.2*j+0.2)), 0:up(shape[1]*0.2*j), :, :] += w[i][0][key][up(shape[0]*0.2*j):up(shape[0]*(0.2*j+0.2)), 0:up(shape[1]*0.2*j), :, :]/num[j]
        elif len(shape) ==2:
          w_avg[key][:, up(shape[1]*0.2*j):up(shape[1]*(0.2*j+0.2))] += w[i][0][key][:, up(shape[1]*0.2*j):up(shape[1]*(0.2*j+0.2))]/num[j]
        else:
          if len(shape)>0 and (shape[0]>10):
            w_avg[key][up(shape[0]*0.2*j):up(shape[0]*(0.2*j+0.2))] += w[i][0][key][up(shape[0]*0.2*j):up(shape[0]*(0.2*j+0.2))]/num[j]
          elif len(shape)==0: #last bias [10] and num_batches_tracked
            if w_avg[key] is None:
              print('None!')
            w_avg[key] =  w[i][0][key]
            #print('==========================wi key', w_avg[key])
          else:
            #print('=======================type of key',key, len(shape) ) ## linear.bias.1
            w_avg[key] += w[i][0][key]/num[0]/(p_index+1)
  for key in w_avg.keys():
    if len(w_avg[key].shape)<=1 and key!='linear.bias':
      w_avg[key] = BN_layer[4][key]
  return w_avg


def FedAvgR(w, BN_layer, args): # [0]: weight, [1]: dropout probability

    w_avg = copy.deepcopy(w[0][0]) # global model
    
    # for k in w_avg.keys():
    #  w_avg[k] = 0*w_avg[k] # model parameter initalization
    
    a = [0]*(len(w)-1) # 10
    num = [len(w)-1]*(len(BN_layer)+1) # 5
    for n in range(1, len(w)):
        # a[n-1] = int(5*w[n][1])
        a[n-1] = w[n][1]
        
    b = [0]*len(BN_layer)
    for i in range(len(b)):
        b[i] = a.count(i)
        num[i+1] = num[i] - b[i]
        
    # b = [a1, ..., a5] = [2, 1, 2, 2, 3] / num = [10, 8, 7, 5, 3]
    # [a1, ..., a5] = [2, 1, 5, 2, 0] / num = [10, 8, 7, 2, 0]
    # print("a:", a, "b:", b, "num:", num)
    # num[4] != 0 -> zeroindex = 4.
    # nun[4] == 0 -> zeroindex = 3.
    zero_idx = 0
    for o in range(1,len(BN_layer)):
        if num[o] == 0:
            break
        zero_idx += 1
    max_p_updated = args.ps[zero_idx] # 0.2*(zero_idx)  # define a maximum size of model that is updated
    # print(max_p_updated)
    for key in w_avg.keys():
        '''
        Initialization
        Non-updated params are not zero-ed and updated params are zero-ed.
        '''
        shape = w_avg[key].shape
        if len(shape) == 4:
            if key == 'conv1.weight':
                w_avg[key][0:up(shape[0]*max_p_updated), :, :, :] = 0*w_avg[key][0:up(shape[0]*max_p_updated), :, :, :]
            else: # other conv layers (layer1.0.conv1.weight -> shape = [16, 16, 3, 3]. layer2.0.conv1.weight -> hsape = [32, 16, 3, 3])
                w_avg[key][0:up(shape[0]*max_p_updated), 0:up(shape[1]*max_p_updated), :, :] = 0*w_avg[key][0:up(shape[0]*max_p_updated), 0:up(shape[1]*max_p_updated), :, :]

        elif len(shape) == 2: # linear.weight, 
            w_avg[key][:, 0:up(shape[1]*max_p_updated)] = 0*w_avg[key][:, 0:up(shape[1]*max_p_updated)]
        
        else: # bn1.weight
            if key == 'linear.bias' or key == 'fc.bias':
                w_avg[key] = 0*w_avg[key]

    list_ps = copy.deepcopy(args.ps)
    list_ps.insert(0, 0)
    for key in w_avg.keys():
        shape = w_avg[key].shape
        for i in range(1, len(w)):
            # index = int((5*w[i][1])-1)
            index = w[i][1]
            for j in range(index+1):
                # li = up(shape[0]*0.2*j) # list_ps[j]
                # hi = up(shape[0]*(0.2*j+0.2)) # list_ps[j+1]
                # li2 = up(shape[1]*0.2*j) # list_ps[j]
                # hi2 = up(shape[1]*(0.2*j+0.2)) # list_ps[j+1]
                if len(shape) > 1:
                    li = up(shape[0]*list_ps[j])
                    hi = up(shape[0]*list_ps[j+1])
                    li2 = up(shape[1]*list_ps[j])
                    hi2 = up(shape[1]*list_ps[j+1])
                    if len(shape) == 4:
                        if key == 'conv1.weight': # w[key][out_ch, in_ch, kernel_size, kernel_size]
                            w_avg[key][li:hi, :, :, :] += w[i][0][key][li:hi, :, :, :]/num[j]
                        else:
                            w_avg[key][li:hi, li2:hi2, :, :] += w[i][0][key][li:hi, li2:hi2, :, :]/num[j]
                            if j!=0:
                                w_avg[key][0:li, li2:hi2, :, :] += w[i][0][key][0:li, li2:hi2, :, :]/num[j]
                                w_avg[key][li:hi, 0:li2, :, :] += w[i][0][key][li:hi, 0:li2, :, :]/num[j]
                    elif len(shape) == 2:
                        w_avg[key][:, li2:hi2] += w[i][0][key][:, li2:hi2]/num[j]
            if key == 'linear.bias' or key == 'fc.bias':
                w_avg[key] += w[i][0][key]/num[0]
                
    for key in w_avg.keys(): # Global BN embedding
            if len(w_avg[key].shape)<=1 and key!='linear.bias' and key!='fc.bias':
                w_avg[key] = BN_layer[-1][key]
                
    return w_avg
  
  
def MAAvg(w, BN_layer, args, commonLayers, singularLayers): # [0]: weight, [1]: dropout probability

    w_avg = copy.deepcopy(w[0][0]) # global model
    
    # for k in w_avg.keys():
    #  w_avg[k] = 0*w_avg[k] # model parameter initalization
    
    a = [0]*(len(w)-1) # 10
    num = [len(w)-1]*(len(BN_layer)+1) # 5
    for n in range(1, len(w)):
        a[n-1] = w[n][1]
        
    b = [0]*len(BN_layer)
    for i in range(len(b)):
        b[i] = a.count(i)
        num[i+1] = num[i] - b[i]
        
    # b = [a1, ..., a5] = [2, 1, 2, 2, 3] / num = [10, 8, 7, 5, 3]
    # [a1, ..., a5] = [2, 1, 5, 2, 0] / num = [10, 8, 7, 2, 0]
    # print("a:", a, "b:", b, "num:", num)
    # num[4] != 0 -> zeroindex = 4.
    # nun[4] == 0 -> zeroindex = 3.
    zero_idx = 0
    for o in range(1,len(BN_layer)):
        if num[o] == 0:
            break
        zero_idx += 1
    max_p_updated = args.ps[zero_idx] # define a maximum size of model that is updated
    # print(max_p_updated)
    '''
      Initialization
      Non-updated params are not zero-ed and updated params are zero-ed.
    '''
    for key in w_avg.keys():
        shape = w_avg[key].shape
        if len(shape) == 4:
            if key == 'conv1.weight':
                w_avg[key][0:up(shape[0]*max_p_updated), :, :, :] = 0*w_avg[key][0:up(shape[0]*max_p_updated), :, :, :]
            else: # other conv layers (layer1.0.conv1.weight -> shape = [16, 16, 3, 3]. layer2.0.conv1.weight -> hsape = [32, 16, 3, 3])
                w_avg[key][0:up(shape[0]*max_p_updated), 0:up(shape[1]*max_p_updated), :, :] = 0*w_avg[key][0:up(shape[0]*max_p_updated), 0:up(shape[1]*max_p_updated), :, :]

        elif len(shape) == 2: # linear.weight, 
            w_avg[key][:, 0:up(shape[1]*max_p_updated)] = 0*w_avg[key][:, 0:up(shape[1]*max_p_updated)]
        
        else: # bn1.weight
            if key == 'linear.bias':
                w_avg[key] = 0*w_avg[key]
                
    ''' Averaging '''
    list_ps = copy.deepcopy(args.ps)
    list_ps.insert(0, 0)
    for key in commonLayers:
        shape = w_avg[key].shape
        for i in range(1, len(w)):
            index = w[i][1]
            for j in range(index+1):
                if len(shape) > 1:
                    li = up(shape[0]*list_ps[j]) 
                    hi = up(shape[0]*list_ps[j+1]) 
                    li2 = up(shape[1]*list_ps[j]) 
                    hi2 = up(shape[1]*list_ps[j+1])
                    if len(shape) == 4:
                        if key == 'conv1.weight': # w[key][out_ch, in_ch, kernel_size, kernel_size]
                            w_avg[key][li:hi, :, :, :] += w[i][0][key][li:hi, :, :, :]/num[j]
                        else:
                            w_avg[key][li:hi, li2:hi2, :, :] += w[i][0][key][li:hi, li2:hi2, :, :]/num[j]
                            if j!=0:
                                w_avg[key][0:li, li2:hi2, :, :] += w[i][0][key][0:li, li2:hi2, :, :]/num[j]
                                w_avg[key][li:hi, 0:li2, :, :] += w[i][0][key][li:hi, 0:li2, :, :]/num[j]
                    elif len(shape) == 2:
                        w_avg[key][:, li2:hi2] += w[i][0][key][:, li2:hi2]/num[j]
            if key == 'linear.bias':
                w_avg[key] += w[i][0][key]/num[0]
    
    
    for key in w_avg.keys():
        shape = w_avg[key].shape
        for i in range(1, len(w)):
            index = w[i][1]
            for j in range(index+1):
                if len(shape) > 1:
                    li = up(shape[0]*list_ps[j]) 
                    hi = up(shape[0]*list_ps[j+1]) 
                    li2 = up(shape[1]*list_ps[j]) 
                    hi2 = up(shape[1]*list_ps[j+1])
                    if len(shape) == 4:
                        if key == 'conv1.weight': # w[key][out_ch, in_ch, kernel_size, kernel_size]
                            w_avg[key][li:hi, :, :, :] += w[i][0][key][li:hi, :, :, :]/num[j]
                        else:
                            w_avg[key][li:hi, li2:hi2, :, :] += w[i][0][key][li:hi, li2:hi2, :, :]/num[j]
                            if j!=0:
                                w_avg[key][0:li, li2:hi2, :, :] += w[i][0][key][0:li, li2:hi2, :, :]/num[j]
                                w_avg[key][li:hi, 0:li2, :, :] += w[i][0][key][li:hi, 0:li2, :, :]/num[j]
                    elif len(shape) == 2:
                        w_avg[key][:, li2:hi2] += w[i][0][key][:, li2:hi2]/num[j]
            if key == 'linear.bias':
                w_avg[key] += w[i][0][key]/num[0]
                
    for key in w_avg.keys(): # Global BN embedding
            if len(w_avg[key].shape)<=1 and key!='linear.bias':
                w_avg[key] = BN_layer[-1][key]
                
    return w_avg