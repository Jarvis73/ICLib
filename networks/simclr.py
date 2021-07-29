import torch.nn as nn

from networks import resnet


class ProjectionHead(nn.Module):
    def __init__(self, opt, in_dim, out_dim, norm_layer=None):
        super(ProjectionHead, self).__init__()
        self.opt = opt
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.linear_layers = []
        if opt.proj_head_mode == 'none':
            pass
        elif opt.proj_head_mode == 'linear':
            self.linear_layers = [
                nn.Linear(in_dim, out_dim, bias=False),
                norm_layer(out_dim)    # [diff] use affine (center and scale)
            ]
        elif opt.proj_head_mode == 'nonlinear':
            for j in range(opt.num_proj_layers):
                if j != opt.num_proj_layers:
                    # for the middle layers, use bias and relu for the output.
                    self.linear_layers.extend([
                        nn.Linear(in_dim, in_dim, bias=True),
                        norm_layer(in_dim),
                        nn.ReLU(inplace=True)
                    ])
                else:
                    self.linear_layers.extend([
                        nn.Linear(in_dim, out_dim, bias=False),
                        norm_layer(out_dim) # [diff] use affine (center and scale)
                    ])
        else:
            raise ValueError

        self.module_list = nn.ModuleList(self.linear_layers)

    def forward(self, x):
        if self.opt.proj_head_mode == 'none':
            return x

        hiddens_list = [x]
        if self.opt.proj_head_mode == 'linear':
            for m in self.module_list:
                x = m(x)
            hiddens_list.append(x)
            return hiddens_list
        elif self.opt.proj_head_mode == 'nonlinear':
            for j in range(self.opt.num_proj_layers):
                for m in self.module_list[j * 3:j * 3 + 3]:
                    x = m(x)
                hiddens_list.append(x)
        else:
            raise ValueError

        return hiddens_list[-1], hiddens_list[self.opt.ft_proj_selector]


class SupervisedHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(SupervisedHead, self).__init__()
        self.linear_layer = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, x):
        return self.linear_layer(x)


class SimCLR(nn.Module):
    def __init__(self, opt, mode, num_classes):
        super(SimCLR, self).__init__()
        self.opt = opt
        self.mode = mode

        resnet_depth = int(opt.backbone[6:])
        cifar_stem = opt.image_size <= 32
        if opt.bn == 'bn':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError(f'batch norm choice {opt.bn} is not implemented')

        self.encoder = resnet.resnet(resnet_depth, cifar_stem, norm_layer, opt)
        self.encoder.maybe_freeze_parameters(mode)

        self.projector = ProjectionHead(opt, self.encoder.out_dim, opt.proj_out_dim)
        if self.mode == 'finetune' or opt.lineareval_while_pretraining:
            self.supervised_head = SupervisedHead(self.encoder.out_dim, num_classes)

    def forward(self, x):
        # Base network forward pass
        hiddens = self.encoder(x).mean(dim=(2, 3))
        # Add heads
        projection_head_outputs, supervised_head_inputs = self.projector(hiddens)

        if self.mode == 'finetune':
            supervised_head_outputs = self.supervised_head(supervised_head_inputs)
            return None, supervised_head_outputs
        elif self.mode == 'pretrain' and self.opt.lineareval_while_pretraining:
            # When performing pretraining and linear evaluation together we do not
            # want information from linear eval flowing back into pretraining network
            # so we put a stop_gradient.
            supervised_head_outputs = self.supervised_head(supervised_head_inputs.detach())
            return projection_head_outputs, supervised_head_outputs
        else:
            return projection_head_outputs, None
