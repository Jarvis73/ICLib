import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 max_iter,
                 power=0.9,
                 lr_end=0,
                 last_epoch=-1):
        super(PolyLR, self).__init__(optimizer, last_epoch)
        self.max_iter = max_iter
        self.power = power
        self.lr_end = lr_end

    def get_lr(self):
        factor = (1 - self.last_epoch / self.max_iter) ** self.power
        factor = max(factor, 0)
        return [(base_lr - self.lr_end) * factor + self.lr_end
                for base_lr in self.base_lrs]


class WarmUpAndCosineDecay(_LRScheduler):
    def __init__(self,
                 opt,
                 optimizer,
                 num_examples,
                 last_epoch=-1):
        self.opt = opt
        self.total_steps = opt.train_steps or (
            num_examples * opt.train_epochs // opt.bs + 1
        )
        self.warmup_steps = self.opt.warmup_epochs * num_examples // self.opt.bs
        super(WarmUpAndCosineDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Apply base learning rate scaling
        if self.opt.lr_scaling == 'linear':
            scaled_lrs = [base_lr * self.opt.bs / 256. for base_lr in self.base_lrs]
        elif self.opt.lr_scaling == 'sqrt':
            scaled_lrs = [base_lr * math.sqrt(self.opt.bs) for base_lr in self.base_lrs]
        else:
            raise ValueError(f'Wrong `lr_scaling`: {self.opt.lr_scaling}')

        # Cosine decay learning rate schedule
        if self.last_epoch < self.warmup_steps:
            factor = self.last_epoch / float(self.warmup_steps)
        else:
            factor = 0.5 * (1 + math.cos((self.last_epoch - self.warmup_steps)
                                         / float(self.total_steps - self.warmup_steps) * math.pi))

        return [base_lr * factor for base_lr in scaled_lrs]


def get(model, opt, num_examples):
    if isinstance(model, list):
        params_group = model
    elif isinstance(model, torch.nn.Module):
        params_group = model.parameters()
    else:
        raise TypeError(f"`model` must be an nn.Model or a list,"
                        f" got {type(model)}")

    # =========================================================================
    # Optimizer
    # =========================================================================
    if opt.optimizer == "sgd":
        optimizer_params = {"momentum": opt.sgd_momentum,
                            "weight_decay": opt.weight_decay,
                            "nesterov": opt.sgd_nesterov}
        optimizer = torch.optim.SGD(params_group, opt.lr, **optimizer_params)
    elif opt.optimizer == "adam":
        optimizer_params = {"betas": (opt.adam_beta1, opt.adam_beta2),
                            "eps": opt.adam_epsilon,
                            "weight_decay": opt.weight_decay}
        optimizer = torch.optim.Adam(params_group, opt.lr, **optimizer_params)
    else:
        raise ValueError("Not supported optimizer: " + opt.opti)

    # =========================================================================
    # Scheduler
    # =========================================================================
    if opt.lrp == "period_step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=opt.lr_step,
            gamma=opt.lr_rate)
    elif opt.lrp == "custom_step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=opt.lr_boundaries,
            gamma=opt.lr_rate)
    elif opt.lrp == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=opt.lr_rate,
            patience=opt.lr_patience,
            threshold=opt.lr_min_delta,
            cooldown=opt.cool_down,
            min_lr=opt.lr_end)
    elif opt.lrp == "cosine":
        max_steps = num_examples * opt.train_epochs // opt.bs + 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_steps,
            eta_min=opt.lr_end)
    elif opt.lrp == "poly":
        max_steps = num_examples * opt.train_epochs // opt.bs + 1
        scheduler = PolyLR(optimizer,
                           max_iter=max_steps,
                           power=opt.power,
                           lr_end=opt.lr_end)
    elif opt.lrp == 'warmup_cosine':
        scheduler = WarmUpAndCosineDecay(
            opt,
            optimizer,
            num_examples)
    else:
        raise ValueError

    return optimizer, scheduler


if __name__ == '__main__':
    import torch.nn as nn
    import matplotlib.pyplot as plt

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = nn.Linear(3, 10)

        def forward(self, x):
            return self.linear(x)

    class Config(object):
        lrp = 'warmup_cosine'
        lr_scaling = 'sqrt'
        bs = 256
        train_epochs = 400
        train_steps = 0
        warmup_epochs = 10
        lr = 0.2
        optimizer = 'sgd'
        sgd_momentum = 0.9
        sgd_nesterov = True
        weight_decay = 1e-4

    opt_ = Config()
    model_ = Model()

    optimizer_, scheduler_ = get(model_, opt_, 60000)

    xs = []
    ys = []
    for x in range(opt_.train_epochs * 60000 // opt_.bs + 1):
        xs.append(x * opt_.bs)
        ys.append(optimizer_.param_groups[0]['lr'])
        scheduler_.step()

    plt.plot(xs, ys)
    plt.show()
