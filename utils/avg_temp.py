import copy
import torch
# from math import ceil as up
from utils.util import up


def FedAvg(w, Norm_layers, Step_layers): # SINGLE MODEL AVERAGING TEST
    w_ref = copy.deepcopy(w[0][0])
    
    for key in w[0][0].keys():
        if not 'norm' in key:
            for uix in range(1,len(w)):
                if uix == 1:
                    w_ref[key] = w[uix][0][key]/10
                else:
                    w_ref[key] += w[uix][0][key]/10
        elif 'norm' in key:
            for uix in range(1,len(w)):
                if uix == 1:
                    Norm_layers[0][key] = w[uix][0][key]/10
                else:
                    Norm_layers[0][key] += w[uix][0][key]/10
        elif 'step' in key:
            for uix in range(1,len(w)):
                if uix == 1:
                    Step_layers[0][key] = w[uix][0][key]/10
                else:
                    Step_layers[0][key] += w[uix][0][key]/10            
            
    
    return w_ref, Norm_layers

def MAAvg(w, BN_layers, Step_layers, args, commonLayers, singularLayers): # [0]: weight, [1]: dropout probability

    w_avg = copy.deepcopy(w[0][0]) # global model
    
    # for k in w_avg.keys():
    #  w_avg[k] = 0*w_avg[k] # model parameter initalization
    
    a = [0]*(len(w)-1) # 10
    acc_num = [len(w)-1]*(args.num_models+1) # [10]*5+1. accumulated number of models.
    b = [0]*args.num_models # 5. number of model idxs for federation.

    for n in range(1, len(w)):
        a[n-1] = w[n][1]
        
    for i in range(len(b)):
        b[i] = a.count(i)
        acc_num[i+1] = acc_num[i] - b[i]
        
    # b = [a1, ..., a5] = [2, 1, 2, 2, 3] / num = [10, 8, 7, 5, 3]
    # [a1, ..., a5] = [2, 1, 5, 2, 0] / num = [10, 8, 7, 2, 0]
    # print("a:", a, "b:", b, "num:", num)
    # num[4] != 0 -> zeroindex = 4.
    # nun[4] == 0 -> zeroindex = 3.
    
    zero_idx = 0
    for o in range(1,args.num_models):
        if acc_num[o] == 0:
            break
        zero_idx += 1
    max_p_updated = args.ps[zero_idx] # define a maximum size of model that is updated
    # print(max_p_updated)
    '''
      Initialization
      Non-updated params are not zero-ed and updated params are zero-ed.
      The case of {parameter # of width[i],depth[i] > parameter # of width[j],depth[j],
      but width[i] < width[j]} for any i, j is not considered.
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
            if key == 'fc.bias':
                w_avg[key] = 0*w_avg[key]   
   
    '''
    BN/Step initialization and Averaged: num_batches_tracked are employed for running mean & var
    '''
    for k in range(len(BN_layers)):
        if b[k] != 0:
            for key in BN_layers[k].keys():
                if 'num_batches_tracked' not in key:
                    BN_layers[k][key] = 0*BN_layers[k][key]
                    
            for key in Step_layers[k].keys():
                Step_layers[k][key] = 0*Step_layers[k][key]
                    
    for i in range(1, len(w)):
        index = w[i][1]
        for key in BN_layers[index].keys():
            if 'num_batches_tracked' not in key:
                BN_layers[index][key] += w[i][0][key]/b[index]
            else:
                BN_layers[index][key] += w[i][0][key]
                
        for key in Step_layers[index].keys():
            Step_layers[index][key] += w[i][0][key]/b[index]
     
                
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
                            w_avg[key][li:hi, :, :, :] += w[i][0][key][li:hi, :, :, :]/acc_num[j]
                        else: # other conv layers
                            w_avg[key][li:hi, li2:hi2, :, :] += w[i][0][key][li:hi, li2:hi2, :, :]/acc_num[j]
                            if j!=0:
                                w_avg[key][0:li, li2:hi2, :, :] += w[i][0][key][0:li, li2:hi2, :, :]/acc_num[j]
                                w_avg[key][li:hi, 0:li2, :, :] += w[i][0][key][li:hi, 0:li2, :, :]/acc_num[j]
                    elif len(shape) == 2: # linear.weight
                        w_avg[key][:, li2:hi2] += w[i][0][key][:, li2:hi2]/acc_num[j]
            if key == 'fc.bias': # independent of j
                w_avg[key] += w[i][0][key]/acc_num[0]
    
    # a = [0]*(len(w)-1) # 10, model indices trained
    # acc_num = [len(w)-1]*(args.num_models+1) # [10]*5+1
    # b = [0]*args.num_models

    # for n in range(1, len(w)):
    #     a[n-1] = w[n][1]
        
    # for i in range(len(b)):
    #     b[i] = a.count(i)
    #     acc_num[i+1] = acc_num[i] - b[i]
        
    for key in singularLayers:
        ''' layerX.Y.~ ResNet18: 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1 '''
        X = int(key.split('.')[0][-1])-1 # indexify
        Y = int(key.split('.')[1])
        # if X == 3 and Y == 1:
            # print('layer4.x')
        
        list_ps = [0]
        active_model_idx = []
        
        for i in range(args.num_models):
            # if args.s2D[args.num_models-1-i][0][X][Y]:
            if args.s2D[i][0][X][Y]:
                list_ps.append(args.ps[i])
                active_model_idx.append(i)
        '''
        Create list_ps with activated model rate = [0, 0.4, 0.6, 0.8, 1]
        '''

        # print(active_model_idx)
        a = [] # model rate trained
        for n in range(1, len(w)):
            if w[n][1] in active_model_idx:
                a.append(args.ps[w[n][1]])
        
        acc_num = [len(a)]*(len(active_model_idx)+1)
        b = [0]*(len(active_model_idx))
        
        j = 0    
        for i in range(len(b)):
            b[j] = a.count(list_ps[i+1])
            acc_num[j+1] = acc_num[j] - b[j]
            j += 1

        '''
        b: the number of each submodel trained
        acc_num: number of a certain parameter of submodel trained (the smallest model)
        
        e.g., 0 non-trained / 1 once / 2 three times / 3 four times / 4 twice
        then, b = [0, 1, 3, 4, 2] and
        acc_num = [10, 10, 9, 6, 2, 0]
        -> 0~0.2: 10 times / 0.2~0.4: 10 times / 0.4~0.6: 9 times / 0.6~0.8: 6 times / 0.8~1: twice
        '''        
        
        shape = w_avg[key].shape
        for i in range(1, len(w)):
            model_index = w[i][1]
            if model_index in active_model_idx:
                index = active_model_idx.index(model_index)
                for j in range(index+1):
                    if len(shape) > 1:
                        li = up(shape[0]*list_ps[j])
                        hi = up(shape[0]*list_ps[j+1])
                        li2 = up(shape[1]*list_ps[j])
                        hi2 = up(shape[1]*list_ps[j+1])
                        if len(shape) == 4:
                            if key == 'conv1.weight': # w[key][out_ch, in_ch, kernel_size, kernel_size]
                                w_avg[key][li:hi, :, :, :] += w[i][0][key][li:hi, :, :, :]/acc_num[j]
                            else:
                                w_avg[key][li:hi, li2:hi2, :, :] += w[i][0][key][li:hi, li2:hi2, :, :]/acc_num[j]
                                if j!=0:
                                    w_avg[key][0:li, li2:hi2, :, :] += w[i][0][key][0:li, li2:hi2, :, :]/acc_num[j]
                                    w_avg[key][li:hi, 0:li2, :, :] += w[i][0][key][li:hi, 0:li2, :, :]/acc_num[j]
                        elif len(shape) == 2:
                            w_avg[key][:, li2:hi2] += w[i][0][key][:, li2:hi2]/acc_num[j]
    

    for key in BN_layers[-1].keys(): # Global BN embedding
        w_avg[key] = copy.deepcopy(BN_layers[-1][key])
                
    return w_avg, BN_layers, Step_layers


def MAAvg_vit(w, Norm_layers, Step_layers, args, commonLayers, singularLayers, local_models): # [0]: weight, [1]: dropout probability

    w_avg = copy.deepcopy(w[0][0]) # global model
    
    # for k in w_avg.keys():
    #  w_avg[k] = 0*w_avg[k] # model parameter initalization
    
    a = [0]*(len(w)-1) # 10
    acc_num = [len(w)-1]*(args.num_models+1) # [10]*5+1. accumulated number of models.
    b = [0]*args.num_models # 5. number of model idxs for federation.

    for n in range(1, len(w)):
        a[n-1] = w[n][1]
    max_model_idx = max(a)
    idx_in_w = a.index(max_model_idx)
    
    for i in range(len(b)):
        b[i] = a.count(i)
        acc_num[i+1] = acc_num[i] - b[i]
        
    # b = [a1, ..., a5] = [2, 1, 2, 2, 3] / num = [10, 8, 7, 5, 3]
    # [a1, ..., a5] = [2, 1, 5, 2, 0] / num = [10, 8, 7, 2, 0]
    # print("a:", a, "b:", b, "num:", num)
    # num[4] != 0 -> zeroindex = 4.
    # nun[4] == 0 -> zeroindex = 3.
    
    zero_idx = 0
    for o in range(1,args.num_models):
        if acc_num[o] == 0:
            break
        zero_idx += 1
    max_p_updated = args.ps[zero_idx] # define a maximum size of model that is updated
    # new_embed_dim = math.ceil(embed_dim*width_ratio/num_heads)*num_heads
    # print(max_p_updated)
    '''
      Initialization
      Non-updated params are not zero-ed and updated params are zero-ed.
      The case of {parameter # of width[i],depth[i] > parameter # of width[j],depth[j],
      but width[i] < width[j]} for any i, j is not considered.
    '''
    for key in w_avg.keys():
        shape = w[idx_in_w][0][key].shape
        if len(shape) == 4: # [0:oi, :, :, :]  patch_embedding.proj.weight
            oi = shape[0]
            w_avg[key][0:oi, :, :, :] = 0*w_avg[key][0:oi, :, :, :]
        elif len(shape) == 3: # cls_token, pos_embed
            oi = shape[2]
            w_avg[key][:,:,0:oi] = 0*w_avg[key][:,:,0:oi]
        elif len(shape) == 2: # attn.qkv.weight, attn.proj.weight, mlp.fc.weight
            ii = shape[1]
            oi = shape[0]
            if key == 'head.weight':
                w_avg[key][:,0:ii] = 0*w_avg[key][:,0:ii]
            else:
                w_avg[key][0:oi,0:ii] = 0*w_avg[key][0:oi,0:ii]
        elif len(shape) == 1:
            if key == 'head.bias':
                w_avg[key] = 0*w_avg[key]
            elif not 'step' in key and not 'norm' in key:
                ii = shape[0]
                w_avg[key][0:ii] = 0*w_avg[key][0:ii]
    
    '''
    BN/Step initialization and Averaged: num_batches_tracked are employed for running mean & var
    '''
    for k in range(len(Norm_layers)):
        if b[k] != 0:
            for key in Norm_layers[k].keys():
                Norm_layers[k][key] = 0*Norm_layers[k][key]
                    
            for key in Step_layers[k].keys():
                Step_layers[k][key] = 0*Step_layers[k][key]
                    
    for i in range(1, len(w)):
        index = w[i][1]
        for key in Norm_layers[index].keys():
            if 'num_batches_tracked' not in key:
                Norm_layers[index][key] += w[i][0][key]/b[index]
            else:
                Norm_layers[index][key] += w[i][0][key]
                
        for key in Step_layers[index].keys():
            Step_layers[index][key] += w[i][0][key]/b[index]
     
    # for i in range(1, len(w)): # for test
    #     # index = w[i][1]
    #     index = 0
    #     for key in Norm_layers[index].keys():
    #         Norm_layers[index][key] += w[i][0][key]/10
                
    #     for key in Step_layers[index].keys():
    #         Step_layers[index][key] += w[i][0][key]/10
    
    # Norm_layers[1] = copy.deepcopy(Norm_layers[0])
    # Step_layers[1] = copy.deepcopy(Step_layers[0])
                
    ''' Averaging '''
    list_ps = copy.deepcopy(args.ps)
    list_ps.insert(0, 0)
    
    # args.width[_] = ceil(args.width[_]*768/num_head)*num_head/768
    
    for key in commonLayers:
    # len(shape)=4: patch_embed.proj.weight
    # len(shape)=3: cls_token, pos_embed
    # len=2: head.weight
    # len=1: patch_embed.proj.bias, head.bias
        shap = w_avg[key].shape
        for i in range(1, len(w)):
            index = w[i][1]
            for j in range(index+1):
                if j==0:
                    shapel = [0, 0, 0, 0]
                    shapeh = local_models[j].state_dict()[key].shape
                else:
                    shapel = local_models[j-1].state_dict()[key].shape
                    shapeh = local_models[j].state_dict()[key].shape
                    
                if len(shap) == 4:
                    li = shapel[0]
                    hi = shapeh[0]
                    w_avg[key][li:hi,:,:,:] += w[i][0][key][li:hi,:,:,:]/acc_num[j]
                elif len(shap) == 3:
                    li = shapel[2]
                    hi = shapeh[2]
                    w_avg[key][:,:,li:hi] += w[i][0][key][:,:,li:hi]/acc_num[j]
                elif len(shap) == 2:
                    li = shapel[1]
                    hi = shapeh[1]
                    w_avg[key][:,li:hi] += w[i][0][key][:,li:hi]/acc_num[j]
                elif len(shap) == 1:
                    if key == 'patch_embed.proj.bias':
                        li = shapel[0]
                        hi = shapeh[0]
                        w_avg[key][li:hi] += w[i][0][key][li:hi]/acc_num[j]
            if key == 'head.bias': ############################
                w_avg[key] += w[i][0][key]/acc_num[0]
    
    # a = [0]*(len(w)-1) # 10, model indices trained
    # acc_num = [len(w)-1]*(args.num_models+1) # [10]*5+1
    # b = [0]*args.num_models

    # for n in range(1, len(w)):
    #     a[n-1] = w[n][1]
        
    # for i in range(len(b)):
    #     b[i] = a.count(i)
    #     acc_num[i+1] = acc_num[i] - b[i]
        
    for key in singularLayers:
        shap = w_avg[key].shape
        # len=2: attn.qkv.weight, attn.proj.weight, mlp.fc1.weight, mlp.fc2.weight
        # len=1: attn.qkv.bias, attn.proj.bias, mlp.fc1.bias, mlp.fc2.bias
        ''' blocks.X.~ '''
        X = int(key.split('.')[1])
                
        list_ps = [0]
        active_model_idx = []
        list_models = []
        
        for i in range(args.num_models):
            if args.s2D[args.num_models-1-i][X]:
                list_ps.append(args.ps[i])
                active_model_idx.append(i)
                list_models.append(local_models[i])
        '''
        Create list_ps with activated model rate = [0, 0.4, 0.6, 0.8, 1]
        '''

        # print(active_model_idx)
        a = [] # model rate trained
        for n in range(1, len(w)):
            if w[n][1] in active_model_idx:
                a.append(args.ps[w[n][1]])
        
        acc_num = [len(a)]*(len(active_model_idx)+1)
        b = [0]*(len(active_model_idx))
        
        j = 0    
        for i in range(len(b)):
            b[j] = a.count(list_ps[i+1])
            acc_num[j+1] = acc_num[j] - b[j]
            j += 1

        # print("a", a, "b", b, "acc_num", acc_num)
                
        for i in range(1, len(w)):
            model_index = w[i][1]
            if model_index in active_model_idx:
                index = active_model_idx.index(model_index)
                for j in range(index+1):
                    if j==0:
                        shapel = [0, 0, 0, 0]
                        shapeh = list_models[j].state_dict()[key].shape
                    else:
                        shapel = list_models[j-1].state_dict()[key].shape
                        shapeh = list_models[j].state_dict()[key].shape
                    
                    if len(shap) == 2:
                        li = shapel[1]
                        hi = shapeh[1]
                        li2 = shapel[0]
                        hi2 = shapeh[0]
                        w_avg[key][li2:hi2,li:hi] += w[i][0][key][li2:hi2,li:hi]/acc_num[j]
                        if j!=0:
                            w_avg[key][0:li2,li:hi] += w[i][0][key][0:li2,li:hi]/acc_num[j]
                            w_avg[key][li2:hi2,0:li] += w[i][0][key][li2:hi2,0:li]/acc_num[j]
                    elif len(shap) == 1:
                        li = shapel[0]
                        hi = shapeh[0]
                        w_avg[key][li:hi] += w[i][0][key][li:hi]/acc_num[j]                      

    for key in Norm_layers[-1].keys(): # Global BN embedding
        w_avg[key] = copy.deepcopy(Norm_layers[-1][key])
                
    return w_avg, Norm_layers, Step_layers


############


def MAAvg_pyvit(w, Norm_layers, Step_layers, args, commonLayers, singularLayers, local_models): # [0]: weight, [1]: dropout probability

    w_avg = copy.deepcopy(w[0][0]) # global model
    
    # for k in w_avg.keys():
    #  w_avg[k] = 0*w_avg[k] # model parameter initalization
    
    a = [0]*(len(w)-1) # 10
    acc_num = [len(w)-1]*(args.num_models+1) # [10]*5+1. accumulated number of models.
    b = [0]*args.num_models # 5. number of model idxs for federation.

    for n in range(1, len(w)):
        a[n-1] = w[n][1]
    max_model_idx = max(a)
    # idx_in_w = a.index(max_model_idx)
    
    for i in range(len(b)):
        b[i] = a.count(i)
        acc_num[i+1] = acc_num[i] - b[i]
        
    # b = [a1, ..., a5] = [2, 1, 2, 2, 3] / num = [10, 8, 7, 5, 3]
    # [a1, ..., a5] = [2, 1, 5, 2, 0] / num = [10, 8, 7, 2, 0]
    # print("a:", a, "b:", b, "num:", num)
    # num[4] != 0 -> zeroindex = 4.
    # nun[4] == 0 -> zeroindex = 3.
    
    zero_idx = 0
    for o in range(1,args.num_models):
        if acc_num[o] == 0:
            break
        zero_idx += 1
    max_p_updated = args.ps[zero_idx] # define a maximum size of model that is updated
    # new_embed_dim = math.ceil(embed_dim*width_ratio/num_heads)*num_heads
    # print(max_p_updated)
    '''
      Initialization
      Non-updated params are not zero-ed and updated params are zero-ed.
      The case of {parameter # of width[i],depth[i] > parameter # of width[j],depth[j],
      but width[i] < width[j]} for any i, j is not considered.
    '''
    for key in w_avg.keys():
        # shape = w[idx_in_w][0][key].shape
        shape = local_models[max_model_idx].state_dict()[key].shape
        if len(shape) == 4: # [0:oi, :, :, :]  patch_embedding.proj.weight
            oi = shape[0]
            w_avg[key][0:oi, :, :, :] = 0*w_avg[key][0:oi, :, :, :]
        elif len(shape) == 3: # class_token, encoder.pos_embedding
            oi = shape[2]
            w_avg[key][:,:,0:oi] = 0*w_avg[key][:,:,0:oi]
        elif len(shape) == 2: # attn.qkv.weight, attn.proj.weight, mlp.fc.weight
            ii = shape[1]
            oi = shape[0]
            if key == 'heads.head.weight':
                w_avg[key][:,0:ii] = 0*w_avg[key][:,0:ii]
            else:
                w_avg[key][0:oi,0:ii] = 0*w_avg[key][0:oi,0:ii]
        elif len(shape) == 1:
            if key == 'heads.head.bias':
                w_avg[key] = 0*w_avg[key]
            elif not 'step' in key and not 'ln' in key:
                ii = shape[0]
                w_avg[key][0:ii] = 0*w_avg[key][0:ii]
    
    '''
    BN/Step initialization and Averaged: num_batches_tracked are employed for running mean & var
    '''
    for k in range(len(Norm_layers)):
        if b[k] != 0:
            for key in Norm_layers[k].keys():
                Norm_layers[k][key] = 0*Norm_layers[k][key]
                    
            for key in Step_layers[k].keys():
                Step_layers[k][key] = 0*Step_layers[k][key]
                    
    for i in range(1, len(w)):
        index = w[i][1]
        for key in Norm_layers[index].keys():
            if 'num_batches_tracked' not in key:
                Norm_layers[index][key] += w[i][0][key]/b[index]
            else:
                Norm_layers[index][key] += w[i][0][key]
                
        for key in Step_layers[index].keys():
            Step_layers[index][key] += w[i][0][key]/b[index]
     
    # for i in range(1, len(w)): # for test
    #     # index = w[i][1]
    #     index = 0
    #     for key in Norm_layers[index].keys():
    #         Norm_layers[index][key] += w[i][0][key]/10
                
    #     for key in Step_layers[index].keys():
    #         Step_layers[index][key] += w[i][0][key]/10
    
    # Norm_layers[1] = copy.deepcopy(Norm_layers[0])
    # Step_layers[1] = copy.deepcopy(Step_layers[0])
                
    ''' Averaging '''
    list_ps = copy.deepcopy(args.ps)
    list_ps.insert(0, 0)
    
    # args.width[_] = ceil(args.width[_]*768/num_head)*num_head/768
    
    for key in commonLayers:
        '''
        len(shape)=4: conv_proj.weight
        len(shape)=3: class_token, encoder.pos_embedding
        len=2: heads.head.weight
        len=1: conv_proj.bias, heads.head.bias
        '''
        shap = w_avg[key].shape
        for i in range(1, len(w)):
            index = w[i][1]
            for j in range(index+1):
                if j==0:
                    shapel = [0, 0, 0, 0]
                    shapeh = local_models[j].state_dict()[key].shape
                else:
                    shapel = local_models[j-1].state_dict()[key].shape
                    shapeh = local_models[j].state_dict()[key].shape
                    
                if len(shap) == 4:
                    li = shapel[0]
                    hi = shapeh[0]
                    w_avg[key][li:hi,:,:,:] += w[i][0][key][li:hi,:,:,:]/acc_num[j] # only output
                elif len(shap) == 3:
                    li = shapel[2]
                    hi = shapeh[2]
                    w_avg[key][:,:,li:hi] += w[i][0][key][:,:,li:hi]/acc_num[j]
                elif len(shap) == 2:
                    li = shapel[1]
                    hi = shapeh[1]
                    w_avg[key][:,li:hi] += w[i][0][key][:,li:hi]/acc_num[j]
                elif len(shap) == 1:
                    if key == 'conv_proj.bias':
                        li = shapel[0]
                        hi = shapeh[0]
                        w_avg[key][li:hi] += w[i][0][key][li:hi]/acc_num[j]
            if key == 'heads.head.bias':
                w_avg[key] += w[i][0][key]/acc_num[0] # only input
    
    # a = [0]*(len(w)-1) # 10, model indices trained
    # acc_num = [len(w)-1]*(args.num_models+1) # [10]*5+1
    # b = [0]*args.num_models

    # for n in range(1, len(w)):
    #     a[n-1] = w[n][1]
        
    # for i in range(len(b)):
    #     b[i] = a.count(i)
    #     acc_num[i+1] = acc_num[i] - b[i]
        
    for key in singularLayers:
        shap = w_avg[key].shape
        '''
        len=2: self_attention_in_proj_weight, self_attention.out_proj.weight, mlp.0/3/weight
        len=1: self_attention_in_proj_bias, self_attention.out_proj.bias , mlp.0/3/.bias
        '''

        ''' encoder.layers.encoder_layer_X.~ '''
        X = int(key.split('.')[2].split('_')[-1])
                
        list_ps = [0]
        active_model_idx = []
        list_models = []
        
        for i in range(args.num_models):
            if args.s2D[args.num_models-1-i][X]:
                list_ps.append(args.ps[i])
                active_model_idx.append(i)
                list_models.append(local_models[i])
        '''
        Create list_ps with activated model rate = [0, 0.4, 0.6, 0.8, 1]
        '''

        # print(active_model_idx)
        a = [] # model rate trained
        for n in range(1, len(w)):
            if w[n][1] in active_model_idx:
                a.append(args.ps[w[n][1]])
        
        acc_num = [len(a)]*(len(active_model_idx)+1)
        b = [0]*(len(active_model_idx))
        
        j = 0    
        for i in range(len(b)):
            b[j] = a.count(list_ps[i+1])
            acc_num[j+1] = acc_num[j] - b[j]
            j += 1

        # print("a", a, "b", b, "acc_num", acc_num)
                
        for i in range(1, len(w)):
            model_index = w[i][1]
            if model_index in active_model_idx:
                index = active_model_idx.index(model_index)
                for j in range(index+1):
                    if j==0:
                        shapel = [0, 0, 0, 0]
                        shapeh = list_models[j].state_dict()[key].shape
                    else:
                        shapel = list_models[j-1].state_dict()[key].shape
                        shapeh = list_models[j].state_dict()[key].shape
                    
                    if len(shap) == 2:
                        li = shapel[1]
                        hi = shapeh[1]
                        li2 = shapel[0]
                        hi2 = shapeh[0]
                        w_avg[key][li2:hi2,li:hi] += w[i][0][key][li2:hi2,li:hi]/acc_num[j]
                        if j!=0:
                            w_avg[key][0:li2,li:hi] += w[i][0][key][0:li2,li:hi]/acc_num[j]
                            w_avg[key][li2:hi2,0:li] += w[i][0][key][li2:hi2,0:li]/acc_num[j]
                    elif len(shap) == 1:
                        li = shapel[0]
                        hi = shapeh[0]
                        w_avg[key][li:hi] += w[i][0][key][li:hi]/acc_num[j]                      

    for key in Norm_layers[-1].keys(): # Global BN embedding
        w_avg[key] = copy.deepcopy(Norm_layers[-1][key])
                
    return w_avg, Norm_layers, Step_layers


###################### HeteroFL ######################################


def NeFedAvg_wo_ic(w, args, commonLayers, singularLayers, bnLayers): # [0]: weight, [1]: dropout probability

    w_avg = copy.deepcopy(w[0][0]) # global model
    
    # for k in w_avg.keys():
    #  w_avg[k] = 0*w_avg[k] # model parameter initalization
    
    a = [0]*(len(w)-1) # 10
    acc_num = [len(w)-1]*(args.num_models+1) # [10]*5+1. accumulated number of models.
    b = [0]*args.num_models # 5. number of model idxs for federation.

    for n in range(1, len(w)):
        a[n-1] = w[n][1]
        
    for i in range(len(b)):
        b[i] = a.count(i)
        acc_num[i+1] = acc_num[i] - b[i]
        
    # b = [a1, ..., a5] = [2, 1, 2, 2, 3] / num = [10, 8, 7, 5, 3]
    # [a1, ..., a5] = [2, 1, 5, 2, 0] / num = [10, 8, 7, 2, 0]
    # print("a:", a, "b:", b, "num:", num)
    # num[4] != 0 -> zeroindex = 4.
    # nun[4] == 0 -> zeroindex = 3.
    
    zero_idx = 0
    for o in range(1,args.num_models):
        if acc_num[o] == 0:
            break
        zero_idx += 1
    max_p_updated = args.ps[zero_idx] # define a maximum size of model that is updated
    # print(max_p_updated)
    '''
      Initialization
      Non-updated params are not zero-ed and updated params are zero-ed.
      The case of {parameter # of width[i],depth[i] > parameter # of width[j],depth[j],
      but width[i] < width[j]} for any i, j is not considered.
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
        
        else:
            if key == 'linear.bias':
                w_avg[key] = 0*w_avg[key]
            else: # bn1.weight
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
                            w_avg[key][li:hi, :, :, :] += w[i][0][key][li:hi, :, :, :]/acc_num[j]
                        else: # other conv layers
                            w_avg[key][li:hi, li2:hi2, :, :] += w[i][0][key][li:hi, li2:hi2, :, :]/acc_num[j]
                            if j!=0:
                                w_avg[key][0:li, li2:hi2, :, :] += w[i][0][key][0:li, li2:hi2, :, :]/acc_num[j]
                                w_avg[key][li:hi, 0:li2, :, :] += w[i][0][key][li:hi, 0:li2, :, :]/acc_num[j]
                    elif len(shape) == 2: # linear.weight
                        w_avg[key][:, li2:hi2] += w[i][0][key][:, li2:hi2]/acc_num[j]
            if key == 'linear.bias': # independent of j (all with same size)
                w_avg[key] += w[i][0][key]/acc_num[0]
                    

    for key in bnLayers:
        shape = w_avg[key].shape
        for i in range(1, len(w)):
            index = w[i][1]
            for j in range(index+1):
                li = up(shape[0]*list_ps[j])
                hi = up(shape[0]*list_ps[j+1])
                w_avg[key][li:hi] += w[i][0][key][li:hi]/acc_num[j]
                            
    # a = [0]*(len(w)-1) # 10, model indices trained
    # acc_num = [len(w)-1]*(args.num_models+1) # [10]*5+1
    # b = [0]*args.num_models

    # for n in range(1, len(w)):
    #     a[n-1] = w[n][1]
        
    # for i in range(len(b)):
    #     b[i] = a.count(i)
    #     acc_num[i+1] = acc_num[i] - b[i]
        
    for key in singularLayers:
        ''' layerX.Y.~ ResNet18: 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1 '''
        X = int(key[5])-1 # indexify
        Y = int(key[7])
        
        # if X == 3 and Y == 1:
            # print('layer4.x')
        
        list_ps = [0]
        active_model_idx = []
        
        for i in range(args.num_models):
            list_ps.append(args.ps[i])
            active_model_idx.append(i)
        '''
        Create list_ps with activated model rate = [0, 0.4, 0.6, 0.8, 1]
        '''

        # print(active_model_idx)
        a = [] # model rate trained
        for n in range(1, len(w)):
            if w[n][1] in active_model_idx:
                a.append(args.ps[w[n][1]])
        
        acc_num = [len(a)]*(len(active_model_idx)+1)
        b = [0]*(len(active_model_idx))
        
        j = 0    
        for i in range(len(b)):
            b[j] = a.count(list_ps[i+1])
            acc_num[j+1] = acc_num[j] - b[j]
            j += 1

        # print("a", a, "b", b, "acc_num", acc_num)
        
        
        shape = w_avg[key].shape
        for i in range(1, len(w)):
            model_index = w[i][1]
            if model_index in active_model_idx:
                index = active_model_idx.index(model_index)            
                for j in range(index+1):
                    if len(shape) > 1:
                        li = up(shape[0]*list_ps[j])
                        hi = up(shape[0]*list_ps[j+1])
                        li2 = up(shape[1]*list_ps[j])
                        hi2 = up(shape[1]*list_ps[j+1])
                        if len(shape) == 4:
                            if key == 'conv1.weight': # w[key][out_ch, in_ch, kernel_size, kernel_size]
                                w_avg[key][li:hi, :, :, :] += w[i][0][key][li:hi, :, :, :]/acc_num[j]
                            else:
                                w_avg[key][li:hi, li2:hi2, :, :] += w[i][0][key][li:hi, li2:hi2, :, :]/acc_num[j]
                                if j!=0:
                                    w_avg[key][0:li, li2:hi2, :, :] += w[i][0][key][0:li, li2:hi2, :, :]/acc_num[j]
                                    w_avg[key][li:hi, 0:li2, :, :] += w[i][0][key][li:hi, 0:li2, :, :]/acc_num[j]
                        elif len(shape) == 2:
                            w_avg[key][:, li2:hi2] += w[i][0][key][:, li2:hi2]/acc_num[j]
                if key == 'linear.bias':
                    w_avg[key] += w[i][0][key]/acc_num[0]     

                
    return w_avg