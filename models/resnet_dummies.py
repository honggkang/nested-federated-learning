class ResNetMp7(nn.Module):
    '''
    Depth, width-varying ResNet with kernel size 7
    '''
    def __init__(self, block, num_blocks, step_size_2d_list, p_drop, learnable_step=True, num_classes=10, width_per_group=64):
        super(ResNetMp7, self).__init__()
        # self.base_width = width_per_group
        self.in_planes = up(64*p_drop)
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, up(64*p_drop), num_blocks[0], step_size_2d_list[0], learnable_step=learnable_step, stride=1, base_width=width_per_group)
        self.layer2 = self._make_layer(block, up(128*p_drop), num_blocks[1], step_size_2d_list[1], learnable_step=learnable_step, stride=2, base_width=width_per_group)
        self.layer3 = self._make_layer(block, up(256*p_drop), num_blocks[2], step_size_2d_list[2], learnable_step=learnable_step, stride=2, base_width=width_per_group)
        self.layer4 = self._make_layer(block, up(512*p_drop), num_blocks[3], step_size_2d_list[3], learnable_step=learnable_step, stride=2, base_width=width_per_group)

        # self.fc = nn.Linear(up(512*block.expansion*p_drop), num_classes)
        self.fc = nn.Linear(self.in_planes, num_classes)

    def _make_layer(self, block, planes, num_blocks, step_size_1d_list, learnable_step, stride, base_width):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        i = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, step_size_1d_list[i], learnable_step=learnable_step, stride=stride, base_width=base_width))
            self.in_planes = planes * block.expansion
            i += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas
    
    
class ResNet_c_DW_7(nn.Module): 
    '''
    Depth, width-varying ResNet designed for CIFAR with kernel size 7
    '''
    def __init__(self, block, num_blocks, step_size_2d_list, p_drop, learnable_step=True, num_classes=10):
        super(ResNet_c_DW_7, self).__init__()
        self.in_planes = up(16*p_drop)

        self.conv1 = nn.Conv2d(3, up(16*p_drop), kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(up(16*p_drop))
        self.layer1 = self._make_layer(block, up(16*p_drop), num_blocks[0], step_size_2d_list[0], learnable_step=learnable_step, stride=1)
        self.layer2 = self._make_layer(block, up(32*p_drop), num_blocks[1], step_size_2d_list[1], learnable_step=learnable_step, stride=2)
        self.layer3 = self._make_layer(block, up(64*p_drop), num_blocks[2], step_size_2d_list[2], learnable_step=learnable_step, stride=2)
        self.fc = nn.Linear(up(64*block.expansion*p_drop), num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, step_size_1d_list, learnable_step, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        i = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, step_size_1d_list[i], learnable_step=learnable_step, stride=stride))
            self.in_planes = planes * block.expansion
            i += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas




################################
# dummies


class ResNetM(nn.Module): # Depth-varying ResNet
    def __init__(self, block, num_blocks, step_size_2d_list, num_classes=10):
        super(ResNetM, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], step_size_2d_list[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], step_size_2d_list[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], step_size_2d_list[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], step_size_2d_list[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, step_size_1d_list, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        i = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, step_size_1d_list[i], stride))
            self.in_planes = planes * block.expansion
            i += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3]) # nn.AdaptiveAvgPool2d((1, 1))
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas


class ResNet_W(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks, p_drop, num_classes=10):
        super(ResNet_W, self).__init__()
        self.in_planes = up(64*p_drop)
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, up(64*p_drop), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, up(128*p_drop), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, up(256*p_drop), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, up(512*p_drop), num_blocks[3], stride=2)
        self.fc = nn.Linear(up(512*block.expansion*p_drop), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas


class ResNet_c_W(nn.Module): # Dropout (or pruned) ResNet [width] for CIFAR-10
    def __init__(self, block, num_blocks, p_drop, num_classes=10):
        super(ResNet_c_W, self).__init__()
        self.in_planes = up(16*p_drop)

        self.conv1 = nn.Conv2d(3, up(16*p_drop), kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(up(16*p_drop))
        self.layer1 = self._make_layer(block, up(16*p_drop), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, up(32*p_drop), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block,up(64*p_drop), num_blocks[2], stride=2)
        self.fc = nn.Linear(up(64*block.expansion*p_drop), num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas


class ResNet_c_D(nn.Module): # Depth-varying ResNet for CIFAR
    def __init__(self, block, num_blocks, step_size_2d_list, num_classes=10):
        super(ResNet_c_D, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], step_size_2d_list[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], step_size_2d_list[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], step_size_2d_list[2], stride=2)
        self.fc = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, step_size_1d_list, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        i = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, step_size_1d_list[i], stride))
            self.in_planes = planes * block.expansion
            i += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas
    
    
def resnet18Mp7(step_size_2d_list, num_classes, p): ## NeFL-WD (kernel 7)
    model = ResNetMp7(BasicBlockM, 
                   [2, 2, 2, 2], step_size_2d_list,
                   p_drop = p, num_classes=num_classes
                   )
    return model


def resnet101Mp(step_size_2d_list, p, learnable_step, num_classes): ## NeFL-WD (kernel 3)
    model = ResNetMp(BottleneckM,
                   [3, 4, 23, 3], step_size_2d_list,
                   p_drop = p, learnable_step=learnable_step, num_classes=num_classes
                   )
    return model



# def resnet110h(num_classes):
#     model = ResNetch(block=BasicBlockH, 
#                    num_blocks=[18, 18, 18],
#                    num_classes=num_classes,
#                    )
#     return model


# def resnet110hf(num_classes, p):
#     model = ResNet_c_HeteroFL(block=BasicBlockH,
#                    num_blocks=[18, 18, 18], p_drop = p,
#                    num_classes=num_classes
#                    )
#     return model

############################################

# def resnet18M(step_size_2d_list, num_classes): ## NeFL parent model
#     model = ResNetM(BasicBlockM,
#                    [2, 2, 2, 2], step_size_2d_list,
#                    num_classes=num_classes
#                    )
#     return model

# def resnet18M(step_size_2d_list, num_classes): ## NeFL parent (global) model
#     model = ResNetMp(BasicBlockM,
#                    [2, 2, 2, 2], step_size_2d_list,
#                    num_classes=num_classes
#                    )
#     return model

# def resnet18wd(step_size_2d_list, num_classes): ## NeFL-WD (kernel 3)
#     model = ResNetMp(BasicBlockM,
#                    [2, 2, 2, 2], step_size_2d_list,
#                    p_drop = 1, learnable_step=True, num_classes=num_classes
#                    )
#     return model

# def resnet18M7(step_size_2d_list, num_classes): ## NeFL parent (global) model
#     model = ResNetM7(BasicBlockM,
#                    [2, 2, 2, 2], step_size_2d_list,
#                    num_classes=num_classes
#                    )
#     return model


# def resnet34M(step_size_2d_list, num_classes): ## MAFL parent model
#     model = ResNetM(BasicBlockM,
#                    [3, 4, 6, 3], step_size_2d_list,
#                    num_classes=num_classes
#                    )
#     return model


# def resnet101M(step_size_2d_list, num_classes): ## MAFL parent model
#     model = ResNetM(BottleneckM,
#                    [3, 4, 23, 3], step_size_2d_list,
#                    num_classes=num_classes
#                    )
#     return model


# def resnet101_2Mp(step_size_2d_list, num_classes, p, learnable_step):
#     model = ResNetMp(BottleneckM,
#                    [3, 4, 23, 3], step_size_2d_list,
#                    p_drop = p, learnable_step=learnable_step, num_classes=num_classes, width_per_group=128
#                    )
#     return model
# Bottleneck, [3, 4, 23, 3]

################################


def resnet56_DW7(step_size_2d_list, p, learnable_step, num_classes): # depth/width
    model = ResNet_c_DW_7(BasicBlockM, 
                   [9, 9, 9], step_size_2d_list,
                   p_drop = p, learnable_step=learnable_step, num_classes=num_classes
                   )
    return model

###############################


##############
# dummies

def resnet18f(num_classes, p): ### FjORD
    """Constructs a ResNet-18 model."""
    model = ResNet_W(block=BasicBlock, 
                   num_blocks=[2, 2, 2, 2], p_drop = p,
                   num_classes=num_classes
                   )
    return model

def resnet34f(num_classes, p): ### FjORD
    """Constructs a ResNet-34 model."""
    model = ResNet_W(block=BasicBlock, 
                   num_blocks=[3, 4, 6, 3], p_drop = p,
                   num_classes=num_classes
                   )
    return model

def resnet56f(num_classes, p): # FjORD
    model = ResNet_c_W(block=BasicBlock, 
                   num_blocks=[9, 9, 9], p_drop = p,
                   num_classes=num_classes
                   )
    return model

def resnet110f(num_classes, p):
    model = ResNet_c_W(block=BasicBlock, 
                   num_blocks=[18, 18, 18], p_drop = p,
                   num_classes=num_classes
                   )
    return model

def resnet56M(step_size_2d_list, num_classes): # only depth, learnable step size
    model = ResNet_c_D(BasicBlockM,
                   [9, 9, 9], step_size_2d_list,
                   num_classes=num_classes,
                   )
    return model

def resnet110M(step_size_2d_list, num_classes):
    model = ResNet_c_D(BasicBlockM, 
                   [18, 18, 18], step_size_2d_list,
                   num_classes=num_classes,
                   )
    return model
