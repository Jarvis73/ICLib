from tqdm import tqdm
import torch
from pathlib import Path
from sacred import Experiment

from config import setup_config, setup_runner
from utils.timer import Timer
from core.model import Model
from core.metrics import Accumulator, Accuracy, TopKCategorialAccuracy
from data_kits import Dataset


ex = Experiment("ICLib", base_dir=Path(__file__).parent, save_git_info=False)
setup_config(ex)


@ex.command
def pretrain(_run, _config):
    opt, logger = setup_runner(ex, _run, _config)

    # Create dataset
    datasets = Dataset(opt, logger, splits=('train',), contrast=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # Create model
    model = Model(opt, logger, _run, datasets, 'pretrain', training=True)
    timer = Timer()
    metrics = 'total_loss contrast_loss contrast_acc contrast_entropy' + \
        'supervised_loss supervised_acc' * opt.lineareval_while_pretraining
    accumulator = Accumulator(**{k: 0. for k in metrics.split()})
    log_str_temp = 'loss: {:.4f} con_loss: {:.4f} con_acc: {:.4f} con_ent: {:.4f}' + \
                   'sup_loss: {:.4f} sup_acc: {:.4f}' * opt.lineareval_while_pretraining + \
                   ' [{:.2f}it/s]'

    start_epoch = 0
    model.train()
    for epoch in range(start_epoch, opt.train_epochs):
        # 1. Training
        tqdmm = tqdm(datasets.train_loader, leave=False)
        for data_i in tqdmm:
            images = [x.to(device) for x in data_i['img']]
            labels = data_i['lab'].to(device)

            with timer.start():
                loss, log_dict = model.step_contrast(images, labels)

                accumulator.update(**log_dict)
                tqdmm.set_description(
                    f"[TRAIN] loss: {loss:.4f} "
                    f"lr: {model.optimizer.param_groups[0]['lr']:g}")

                model.step_lr()

        # 2. Log results
        log_str = f"[{epoch + 1:3d}/{opt.train_epochs}] " \
                  f"lr: {model.optimizer.param_groups[0]['lr']:3g} "
        res = accumulator.mean(metrics.split())
        log_str += log_str_temp.format(*res, timer.cps)
        logger.info(log_str)
        for k, v in accumulator.mean(metrics.split(), dic=True).items():
            _run.log_scalar(k, float(v), epoch + 1)

        # 3. save model and reset
        model.save(epoch)
        model.step_lr(epoch_end=True)
        timer.reset()
        accumulator.reset()


@ex.command
def train_then_eval(_run, _config):
    opt, logger = setup_runner(ex, _run, _config)

    # Create dataset
    datasets = Dataset(opt, logger, splits=('train', 'val'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # Create and restore model
    model = Model(opt, logger, _run, datasets, 'finetune', training=True)
    model.try_restore_from_checkpoint()

    timer = Timer()
    metrics = 'loss acc'
    accumulator = Accumulator(**{k: 0. for k in metrics.split()})
    log_str_temp = 'loss: {:.4f} acc: {:.4f} val_loss: {:.4f} top1: {:.4f} top5: {:.4f} ' \
                   '[{:.2f}it/s]'

    start_epoch = 0
    for epoch in range(start_epoch, opt.train_epochs):
        # 1. Training
        model.train()
        tqdmm = tqdm(datasets.train_loader, leave=False)
        for data_i in tqdmm:
            images = data_i['img'].to(device)
            labels = data_i['lab'].to(device)

            with timer.start():
                loss, log_dict = model.step(images, labels)

                accumulator.update(**log_dict)
                tqdmm.set_description(
                    f"[TRAIN] loss: {loss:.4f} "
                    f"lr: {model.optimizer.param_groups[0]['lr']:g}")

                model.step_lr()
        train_res = accumulator.mean(metrics.split())

        # 3. Validation
        model.eval()
        loss = Accumulator(loss=0.)
        top1 = Accuracy()
        top5 = TopKCategorialAccuracy(k=5)
        tqdmm_val = tqdm(datasets.val_loader, leave=False)
        for data_i in tqdmm_val:
            images = data_i['img'].to(device)
            labels = data_i['lab'].to(device)

            with torch.no_grad():
                loss_val, logits = model.test_step(images, labels)
                loss.update(loss=loss_val)
                top1.update(logits, labels)
                top5.update(logits, labels)
        val_res = [loss.mean('loss'), top1.result(), top5.result()]

        # 3. Log results
        for k, v in accumulator.mean(metrics.split(), dic=True).items():
            _run.log_scalar(k, float(v), epoch + 1)
        _run.log_scalar('val_loss', val_res[0], epoch + 1)
        _run.log_scalar('val_top1', val_res[1], epoch + 1)
        _run.log_scalar('val_top5', val_res[2], epoch + 1)

        log_str = f"[{epoch + 1:3d}/{opt.train_epochs}] " \
                  f"lr: {model.optimizer.param_groups[0]['lr']:5g} "
        log_str += log_str_temp.format(*train_res, *val_res, timer.cps)
        logger.info(log_str)

        # 4. save model and reset
        model.save(epoch)
        model.step_lr(epoch_end=True)
        timer.reset()
        accumulator.reset()


@ex.command
def test(_run, _config):
    opt, logger = setup_runner(ex, _run, _config)

    # Create dataset
    datasets = PairDataset(opt, logger, splits=('val',))
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # Create model
    model = Model(opt, logger, _run, isTrain=False)
    timer = Timer()
    model.eval(logger)

    # Validate
    with torch.no_grad():
        tqdmm = tqdm(datasets.val_loader, leave=False)
        for data_i in tqdmm:
            with timer.start():
                images = data_i["img"].to(device)
                labels = data_i["lab"].numpy()

                prob = model.test_step(images)
                pred = prob.argmax(1).cpu().numpy()
            running_metrics.update(labels, pred)

    # Record results
    score, class_acc = running_metrics.get_scores()
    for k, v in score.items():
        logger.info(f'{k}: {v}')
        _run.log_scalar(k, float(v), 0)
    for k, v in class_acc.items():
        logger.info(f'class{k}: {v:.4f}')
        _run.log_scalar(f"class{k}", float(v), 0)

    print_str = f"Mean Acc: {score['Mean Acc']:.4f} Speed: {timer.cps:.2f}it/s"
    logger.info(print_str)

    return f"Mean Acc: {score['Mean Acc']:.4f}"


if __name__ == "__main__":
    ex.run_commandline()
