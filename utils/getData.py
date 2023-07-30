import numpy as np
from torchvision import datasets, transforms
from math import sqrt

# import lasagne
import pickle
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def mnist_iid(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users, seed):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(seed)

    num_shards, num_imgs = 200, 300 # 2 (class) x 100 (users), 2 x 300 (imgs) for each client
    # {0: 5923, 1: 6742, 2: 5958, 3: 6131, 4: 5842, 5: 5421, 6: 5918, 7: 6265, 8: 5851, 9: 5949}
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy() # targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] # label-wise sort
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return

def cifar_iid(dataset, num_users, seed):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(seed)
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(args, dataset):
    """
    Sample non-I.I.D client data from CIFAR dataset 50000
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(args.rs)

    num_shards, num_imgs = args.num_users * args.class_per_each_client, int(50000/args.num_users/args.class_per_each_client)
    # {0: 5000, 1: 5000, 2: 5000, 3: 5000, 4: 5000, 5: 5000, 6: 5000, 7: 5000, 8: 5000, 9: 5000}
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(args.num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] # label-wise sort
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(args.num_users):
        rand_set = set(np.random.choice(idx_shard, args.class_per_each_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

# def load_validation_data(data_folder, mean_image, img_size=32):
#     test_file = os.path.join(data_folder, 'val_data')        

#     d = unpickle(test_file)
#     x = d['data']
#     y = d['labels']
#     x = x / np.float32(255)

#     # Labels are indexed from 1, shift it so that indexes start at 0
#     y = np.array([i-1 for i in y])

#     # Remove mean (computed from training data) from images
#     x -= mean_image

#     img_size2 = img_size * img_size

#     x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
#     x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

#     return dict(
#         X_test=lasagne.utils.floatX(x),
#         Y_test=y.astype('int32'))


# def load_databatch(data_folder, idx, img_size=32):
#     data_file = os.path.join(data_folder, 'train_data_batch_')

#     d = unpickle(data_file + str(idx))
#     x = d['data']
#     y = d['labels']
#     mean_image = d['mean']

#     x = x/np.float32(255)
#     mean_image = mean_image/np.float32(255)

#     # Labels are indexed from 1, shift it so that indexes start at 0
#     y = [i-1 for i in y]
#     data_size = x.shape[0]

#     x -= mean_image

#     img_size2 = img_size * img_size

#     x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
#     x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

#     # create mirrored images
#     X_train = x[0:data_size, :, :, :]
#     Y_train = y[0:data_size]
#     X_train_flip = X_train[:, :, :, ::-1]
#     Y_train_flip = Y_train
#     X_train = np.concatenate((X_train, X_train_flip), axis=0)
#     Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

#     return dict(
#         X_train=lasagne.utils.floatX(X_train),
#         Y_train=Y_train.astype('int32'),
#         mean=mean_image)
    

def getDataset(args):
    if args.dataset =='cifar10':
        ## CIFAR
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), # transforms.Resize(256), transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('/home/hong/NeFL/.data/cifar', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('/home/hong/NeFL/.data/cifar', train=False, download=True, transform=transform_test)

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

    ### downsampled ImageNet
    # imagenet_data = datasets.ImageNet('/home')
        
    ### Flowers
    # tranform_train = transforms.Compose([
    #                                     #   transforms.RandomRotation(30),
    #                                     #   transforms.RandomResizedCrop(224),
    #                                       transforms.RandomHorizontalFlip(),
    #                                       transforms.ToTensor(), 
    #                                       transforms.Normalize([0.485, 0.456, 0.406], 
    #                                                            [0.229, 0.224, 0.225])
    #                                      ])
        
    # tranform_test = transforms.Compose([
    #                                     #   transforms.Resize(256),
    #                                     #   transforms.CenterCrop(224),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize([0.485, 0.456, 0.406], 
    #                                                            [0.229, 0.224, 0.225])
    #                                      ])

    # dataset_train = datasets.Flowers102('/home/hong/NeFL/.data/flowers102', download=True, transform=tranform_train)
    # dataset_test = datasets.Flowers102('/home/hong/NeFL/.data/flowers102', split='test', download=True, transform=tranform_test)
    # split='train',

    ### Food 101
    # tranform_train = transforms.Compose([transforms.RandomRotation(30),
    #                                        transforms.RandomResizedCrop(224),
    #                                        transforms.RandomHorizontalFlip(),ImageNetPolicy(),
    #                                        transforms.ToTensor(),
    #                                        transforms.Normalize([0.485, 0.456, 0.406],
    #                                                             [0.229, 0.224, 0.225])])

    # tranform_test = transforms.Compose([transforms.Resize(255),
    #                                       transforms.CenterCrop(224),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize([0.485, 0.456, 0.406],
    #                                                            [0.229, 0.224, 0.225])])
    # dataset_train = datasets.Food101('/home/hong/NeFL/.data/food101', split='train', download=True, transform=tranform_train)
    # dataset_test = datasets.Food101('/home/hong/NeFL/.data/food101', split='test', download=True, transform=tranform_test)
    return dataset_train, dataset_test

def get_submodel_info(args):
    if args.model_name == 'resnet56':
        if args.method == 'W':
            ps = [sqrt(0.2), sqrt(0.4), sqrt(0.6), sqrt(0.8), 1]
            s2D = [ # full 56  NeFL-W
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ]
            ]
        elif args.method =='WD':
            ps = [sqrt(0.464938491), sqrt(0.607811604), sqrt(0.777135864), sqrt(0.902901024), 1] # resnet 56 (MA4)
            s2D = [ # 56 MA5 NeFL-WD
                        [ [[1, 1, 1, 1, 0, 0, 0, 0, 0] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 0, 0, 0] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 0, 0] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 0] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ]
                ]
        elif args.method=='DD':
            ps = [1, 1, 1, 1, 1]
            s2D = [ # 56 DDa
                        [ [[1, 1, 0, 0, 0, 0, 0, 0, 0] for _ in range(3)] ], # 0.20223
                        [ [[1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0], [1,1,1,1,0,0,0,0,0]] ], # 0.40293
                        [ [[1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0], [1,1,1,1,1,1,0,0,0]] ], # 0.60363
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]] ], # 80478
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ]
                        ]
            # s2D = [ # 56 DD-2
            #     [ [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 0], [1,0,0,0,0,0,0,0,0]] ], # 0.19735
            #     [ [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0], [1,1,1,0,0,0,0,0,0]] ], # 0.39258
            #     [ [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 0, 0], [1,1,1,1,1,0,0,0,0]] ], # 0.60956
            #     [ [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]] ], # 0.80478
            #     [ [[1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ]
            #     ]
    elif args.model_name == 'resnet110':
        if args.method == 'W':
            ps = [sqrt(0.2), sqrt(0.4), sqrt(0.6), sqrt(0.8), 1]
            s2D = [ # full
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ]
                ]
        elif args.method == 'W04':
            ps = [0.2, 0.4, 0.6, 0.8, 1]
            s2D = [ # full
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ]
                ]            
        elif args.method == 'WD':
            ps = [sqrt(0.457252561), sqrt(0.603830987), sqrt(0.774235493), sqrt(0.901429773), 1] # resnet110 MA5
            s2D = [ # 110 MA5 230304 
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ]
                ]
        elif args.method=='DD':
            ps = [1, 1, 1, 1, 1]
            s2D = [
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ],  # 0.20198
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ],  # 0.4
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]] ],  # 0.601
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]] ],  # 0.80185
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] ]
                ]
        elif args.method == 'WD04':
            ps = [sqrt(0.188365555), sqrt(0.365802049), sqrt(0.59388641), sqrt(0.769953979), 1] # resnet110 MA5
            s2D = [ # 110 MA5 230304 
                        [ [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0] for _ in range(3)] ],
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(3)] ]
                ]
        elif args.method == 'DD04':
            ps = [1, 1, 1, 1, 1]
            s2D = [
                        [ [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(3)] ],  # 0.04357
                        [ [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ],  # 0.1615
                        [ [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ],  # 0.36769
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]] ],  # 0.64889
                        [ [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] ]
                ]            
    elif args.model_name == 'resnet18':
        if args.method == 'DD':
            ps = [1, 1, 1, 1, 1] # NEFL ADD
            s2D = [ # 18
                    [ [[1, 1], [0, 0], [1, 1], [0, 0]] ], # 0.2
                    [ [[1, 1], [1, 1], [0, 0], [1, 0]] ], # 0.39
                    [ [[1, 1], [1, 1], [1, 1], [1, 0]] ], # 0.57
                    [ [[1, 0], [1, 1], [0, 0], [1, 1]] ], # 0.8
                    [ [[1, 1], [1, 1], [1, 1], [1, 1]] ]
                ]
        elif args.method == 'OD':
            ps = [1, 1, 1, 1, 1] # NEFL AOD
            s2D = [ # 18
                    [ [[1, 3], [0, 0], [1, 3], [0, 0]] ], # 0.2
                    [ [[1, 1], [1, 3], [0, 0], [2, 0]] ], # 0.39
                    [ [[1, 1], [1, 1], [1, 1], [2, 0]] ], # 0.57
                    [ [[2, 0], [1, 3], [0, 0], [1, 1]] ], # 0.8
                    [ [[1, 1], [1, 1], [1, 1], [1, 1]] ]
                ]            
        elif args.method == 'W':
            ps = [sqrt(0.2), sqrt(0.4), sqrt(0.6), sqrt(0.8), 1] # NEFL W
            s2D = [ # 18
                    [ [[1, 1], [1, 1], [1, 1], [1, 1]] ], # 1
                    [ [[1, 1], [1, 1], [1, 1], [1, 1]] ], # 1
                    [ [[1, 1], [1, 1], [1, 1], [1, 1]] ], # 1
                    [ [[1, 1], [1, 1], [1, 1], [1, 1]] ], # 1
                    [ [[1, 1], [1, 1], [1, 1], [1, 1]] ]
                ]
        elif args.method == 'WD':
            ps = [sqrt(0.34), sqrt(0.4), sqrt(0.6), sqrt(0.8), 1] # NEFL WD
            s2D = [ # 18
                    [ [[1, 1], [1, 1], [1, 1], [1, 0]] ], 
                    [ [[1, 1], [1, 1], [1, 1], [1, 1]] ], 
                    [ [[1, 1], [1, 1], [1, 1], [1, 1]] ],
                    [ [[1, 1], [1, 1], [1, 1], [1, 1]] ], 
                    [ [[1, 1], [1, 1], [1, 1], [1, 1]] ]
                ]
    elif args.model_name == 'resnet34':
        if args.method == 'W':
            ps = [sqrt(0.2), sqrt(0.4), sqrt(0.6), sqrt(0.8), 1]
            s2D = [ # D = [1, 1, 1, 1, 1]
                    [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1]] ],
                    [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1]] ],
                    [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1]] ],
                    [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1]] ],
                    [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1]] ]
                ]
        elif args.method == 'DD':
            ps = [1, 1, 1, 1, 1]
            s2D = [ # D = [0.230670158, 0.39016361, 0.611896062, 0.805875884, 1]
                    [ [[1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0]] ], #             [ [[1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0]] ], # 0.2198601
                    [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0, 0, 1], [1, 0, 0]] ],
                    [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0, 0, 1], [1, 0, 1]] ],
                    [ [[1, 1, 1], [1, 0, 0, 1], [1, 1, 0, 0, 0, 1], [1, 1, 1]] ],
                    [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1]] ]
                ]
        elif args.method == 'OD':
            ps = [1, 1, 1, 1, 1]
            s2D = [ # D = [0.230670158, 0.39016361, 0.611896062, 0.805875884, 1]
                    [ [[3, 0, 0], [4, 0, 0, 0], [6, 0, 0, 0, 0, 0], [3, 0, 0]] ], #             [ [[1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0]] ], # 0.2198601
                    [ [[1, 1, 1], [1, 1, 1, 1], [1, 4, 0, 0, 0, 1], [3, 0, 0]] ],
                    [ [[1, 1, 1], [1, 1, 1, 1], [1, 4, 0, 0, 0, 1], [2, 0, 1]] ],
                    [ [[1, 1, 1], [3, 0, 0, 1], [1, 4, 0, 0, 0, 1], [1, 1, 1]] ],
                    [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1]] ]
                ]    
        elif args.method == 'WD':
            ps = [sqrt(0.378296187), sqrt(0.625390289), sqrt(0.770943105), sqrt(0.899800797), 1]
            s2D = [ # D = [0.52868627, 0.639600594, 0.778267548, 0.889085676, 1]
                    [ [[1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0, 0, 1], [1, 0, 1]] ],
                    [ [[1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 0, 0, 1], [1, 0, 1]] ],
                    [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 0, 1], [1, 0, 1]] ],
                    [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0, 0, 1], [1, 1, 1]] ],
                    [ [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1]] ]
                ]
    return ps, s2D
        