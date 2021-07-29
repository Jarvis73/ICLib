import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from utils.loggers import C as CC
from config import get_rundir
from core import solver
from core import losses as loss_kits
from networks import SimCLR


class Model(object):
    def __init__(self, opt, logger, run, datasets, mode, training=True):
        self.opt = opt
        self.logger = logger
        self.run = run
        self.training = training
        self.mode = mode

        self.num_classes = datasets.train_dataset.num_classes
        self.best_acc = -1

        # Define logdir for saving checkpoints
        self.do_ckpt = False if self.run._id is None else True
        self.logdir = Path(get_rundir(opt, run))
        if self.training:
            self.logdir.mkdir(parents=True, exist_ok=True)

        if opt.model_name == "simclr":
            model = SimCLR(opt, mode, self.num_classes)
        else:
            raise NotImplementedError(f"`{opt.model_name}` is not implemented. []")
        logger.info('The backbone is ' +
                    CC.c(f'{model.__class__.__name__} ({opt.backbone})', CC.BOLD))

        self.model, self.model_DP = self.init_device(model)

        if self.training:
            # Define optimizer and scheduler
            self.optimizer, self.scheduler = solver.get(self.model, opt, len(datasets.train_dataset))
            self.do_step_lr = self.opt.lrp in ["cosine", "poly", "warmup_cosine"]

    def step(self, x, y):
        log_dict = {}

        self.optimizer.zero_grad()
        _, supervised_head_outputs = self.model_DP(x)
        outputs = supervised_head_outputs

        # loss
        loss = loss_kits.supervised_loss(outputs, y)
        log_dict['loss'] = loss.item()

        # acc
        acc = (outputs.argmax(1) == y).float().mean()
        log_dict['acc'] = acc.item()

        loss.backward()
        self.optimizer.step()

        return loss.item(), log_dict

    def test_step(self, x, y):
        _, supervised_head_outputs = self.model_DP(x)
        outputs = supervised_head_outputs

        loss = loss_kits.supervised_loss(outputs, y)
        return loss.item(), outputs

    def step_contrast(self, x, y):
        self.optimizer.zero_grad()
        losses = []
        log_dict = {}

        # forward
        pair_x = torch.cat(x, 0)    # [2*B, 3, H, W]
        projection_head_outputs, supervised_head_outputs = self.model_DP(pair_x)

        if projection_head_outputs is not None:
            outputs = projection_head_outputs
            # loss
            con_loss, logits_con, labels_con = loss_kits.contrastive_loss(
                outputs,
                hidden_norm=self.opt.hidden_norm,
                temperature=self.opt.temperature
            )
            losses.append(con_loss)
            log_dict['contrast_loss'] = con_loss.item()

            # acc
            contrast_acc_val = logits_con.argmax(1) == labels_con
            contrast_acc_val = contrast_acc_val.float().mean()
            log_dict['contrast_acc'] = contrast_acc_val.item()

            # entropy
            logp = F.log_softmax(logits_con, 1)
            p = F.softmax(logits_con, 1)
            entropy_con = -(p * logp).sum() / p.shape[0]
            log_dict['contrast_entropy'] = entropy_con.item()

        if supervised_head_outputs is not None:
            outputs = supervised_head_outputs
            # loss
            if self.mode == 'pretrain' and self.opt.lineareval_while_pretraining:
                y = torch.cat([y, y], 0)
            sup_loss = loss_kits.supervised_loss(outputs, y)
            losses.append(sup_loss)
            log_dict['supervised_loss'] = sup_loss.item()

            # acc
            label_acc = (outputs.argmax(1) == y).float().mean()
            log_dict['supervised_acc'] = label_acc.item()

        loss = sum(losses)
        log_dict['total_loss'] = loss

        # backward
        loss.backward()
        self.optimizer.step()

        return loss.item(), log_dict

    def init_device(self, net: nn.Module):
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
        net_DP = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        return net, net_DP

    def train(self):
        self.model.train()
        self.model_DP.train()

    def eval(self):
        self.model.eval()
        self.model_DP.eval()

    def step_lr(self, epoch_end=False):
        """
        Update learning rate by the specified learning rate policy.
        For 'cosine' and 'poly' policies, the learning rate is updated
        step-by-step. For other policies, the learning rate is updated
        epoch-by-epoch.
        """
        if (not epoch_end and self.do_step_lr) \
                or (epoch_end and not self.do_step_lr):
            self.scheduler.step()

    def save(self, epoch, save_path=None):
        state = self.model.state_dict()
        torch.save(state, str(save_path or self.logdir / 'ckpt.pth'))

    def try_restore_from_checkpoint(self):
        if self.opt.ckpt_id > 0:
            ckpt_path = self.logdir.parent / str(self.opt.ckpt_id) / 'ckpt.pth'
        elif self.opt.ckpt:
            ckpt_path = Path(self.opt.ckpt)
        else:
            ckpt_path = self.logdir / 'ckpt.pth'

        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint not found in {ckpt_path}')

        ckpt = torch.load(str(ckpt_path))
        self.model.load_state_dict(ckpt, strict=False)

        if self.opt.model_name == 'simclr' and self.opt.zero_init_logits_layer:
            for v in self.model.supervised_head.parameters():
                nn.init.zeros_(v)

        self.logger.info(CC.c(f"Checkpoint loaded from {ckpt_path}", CC.BOLD))
