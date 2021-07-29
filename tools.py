import re
from pathlib import Path

import torch
from sacred import Experiment

ex = Experiment("Tools", save_git_info=False)

@ex.config
def config():
    ckpt = ''


@ex.command
def rectify_pretrained_names(_log, ckpt):
    checkpoint = torch.load(ckpt, map_location='cpu')['resnet']

    layer0_map = {
        'net.0.0.weight':                   'encoder.conv1.weight',
        'net.0.1.0.weight':                 'encoder.bn1.weight',
        'net.0.1.0.bias':                   'encoder.bn1.bias',
        'net.0.1.0.running_mean':           'encoder.bn1.running_mean',
        'net.0.1.0.running_var':            'encoder.bn1.running_var',
        'net.0.1.0.num_batches_tracked':    'encoder.bn1.num_batches_tracked'
    }

    backbone_pattern = re.compile(r'net\.(\d)\.blocks\.(\d)\.net\.(\d)[\.\d]+([a-z_]+)')
    skip_layer_pattern = re.compile(r'net\.(\d)\.blocks\.(\d)\.projection\.([a-z]+)\.[\.\d]*([a-z]+)')

    new_state_dict = {}
    for k, v in checkpoint.items():
        if k in layer0_map:
            new_state_dict[layer0_map[k]] = v
            continue

        match = backbone_pattern.search(k)
        if match is not None:
            block_num, layer_num, module_num, module_param = match.groups()
            new_module_num = int(module_num) // 2 + 1
            if int(module_num) % 2 == 0:
                module_name = f'conv{new_module_num}'
            else:
                module_name = f'bn{new_module_num}'
            new_key = f'encoder.layer{block_num}.{layer_num}.{module_name}.{module_param}'
            new_state_dict[new_key] = v

        match = skip_layer_pattern.search(k)
        if match is not None:
            block_num, layer_num, module_name, module_param = match.groups()
            if module_name == 'shortcut':
                module_num = 0
            else:
                module_num = 1
            new_key = f'encoder.layer{block_num}.{layer_num}.downsample.{module_num}.{module_param}'
            new_state_dict[new_key] = v

    ckpt = Path(ckpt)
    new_ckpt = ckpt.parent / f'{ckpt.stem}_renamed.pth'
    torch.save(new_state_dict, new_ckpt)
    _log.info(f"Saved to {new_ckpt}")


if __name__ == '__main__':
    ex.run_commandline()
