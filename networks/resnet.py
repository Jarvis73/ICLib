import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, cifar_stem=False, norm_layer=None, opt=None):
        super(ResNet, self).__init__()
        self.name_ = self.__class__.__name__
        self.opt = opt
        self.inplanes = 64
        self.out_dim = 512 if block == BasicBlock else 2048
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if cifar_stem:
            self.conv1 = nn.Conv2d(3, self.inplanes, 3, stride=1, padding=1, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, 7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        kwargs = {}
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, **kwargs)
        self.init_weight()

    def _make_layer(self, block, planes, blocks, stride=1):
        kwargs = {}
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                norm_layer(planes * block.expansion))

        layers = [block(self.inplanes, planes, stride, downsample=downsample, **kwargs)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if hasattr(self, 'maxpool'):
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _flat_list(self, lst):
        new_lst = []
        for x in lst:
            if isinstance(x, list):
                new_lst.extend(x)
            else:
                new_lst.append(x)
        return new_lst

    def maybe_freeze_parameters(self, mode):
        freeze_layers = [
            [self.conv1, self.bn1], self.layer1, self.layer2, self.layer3, self.layer4
        ]
        if mode == 'finetune' and self.opt.fine_tune_after_block >= 0:
            freeze_layers = freeze_layers[:self.opt.fine_tune_after_block + 1]

        for v in self._flat_list(freeze_layers):
            v.requires_grad = False


def resnet(resnet_depth,
           cifar_stem=False,
           norm_layer=None,
           opt=None):
    """Returns the ResNet backbone model for a given size"""
    model_params = {
        18: {'block': BasicBlock, 'layers': [2, 2, 2, 2]},
        34: {'block': BasicBlock, 'layers': [3, 4, 6, 3]},
        50: {'block': Bottleneck, 'layers': [3, 4, 6, 3]},
        101: {'block': Bottleneck, 'layers': [3, 4, 23, 3]},
        152: {'block': Bottleneck, 'layers': [3, 8, 36, 3]},
        200: {'block': Bottleneck, 'layers': [3, 24, 36, 3]}
    }

    if resnet_depth not in model_params:
        raise ValueError('Not a valid resnet_depth:', resnet_depth)

    params = model_params[resnet_depth]
    return ResNet(
        params['block'],
        params['layers'],
        cifar_stem=cifar_stem,
        norm_layer=norm_layer,
        opt=opt
    )
