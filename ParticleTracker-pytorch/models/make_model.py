import torch
from torch import nn

from models.particle_detector import ParticleDetector
from models.particle_matcher import ParticleMatcher
from models.particle_tracker import ParticleTrackerV1


def _module_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            if m.weight.requires_grad:
                nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                if m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0.)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.weight.requires_grad:
                nn.init.constant_(m.weight, 1.)
            if m.bias.requires_grad:
                nn.init.constant_(m.bias, 0.)


def pd_model_preparation(opt, train=True):
    mode = 'Train' if train else 'Evaluate'
    pd_model = ParticleDetector(
        img_channels=1,
        cell_size=opt.cell_size,
        basic_pow=opt.basic_pow,
        max_num_in_cell=opt.max_num_in_cell,
        dense_div=opt.dense_div,
        conv_num_in_block=opt.conv_num_in_block,
        conv_norm=opt.conv_norm,
        dropout_rate=opt.dropout_rate,
    )
    if train:
        if opt.particle_detector_checkpoint_path is None:
            _module_init(pd_model)
        else:
            pd_model.load_state_dict(torch.load(opt.particle_detector_checkpoint_path, map_location='cpu'))
    else:
        if opt.particle_detector_checkpoint_path is None:
            raise ValueError('No checkpoints file. Can not do evaluation!')
        else:
            pd_model.load_state_dict(torch.load(opt.particle_detector_checkpoint_path, map_location='cpu'))

    if not opt.force_cpu:
        device = torch.device(f'cuda:{opt.cuda_ids[0]}' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    pd_model.to(device)
    if 'cuda' in str(device):
        pd_model = torch.nn.DataParallel(pd_model, device_ids=opt.cuda_ids)
        print(f'{mode} on: {device}, with device_ids: {opt.cuda_ids}')
    else:
        print(f'{mode} on: {device}')

    return pd_model, device


def pt_model_preparation(opt, train=True):
    mode = 'Train' if train else 'Evaluate'

    pd_model = ParticleDetector(
        img_channels=1,
        cell_size=opt.cell_size,
        basic_pow=opt.basic_pow,
        max_num_in_cell=opt.max_num_in_cell,
        dense_div=opt.dense_div,
        conv_num_in_block=opt.conv_num_in_block,
        conv_norm=opt.conv_norm,
        dropout_rate=opt.dropout_rate,
    )

    pm_model = ParticleMatcher(
        desc_pow=opt.desc_pow,
        gnn_layer_pairs=opt.gnn_layer_pairs,
        sinkhorn_iters=opt.sinkhorn_iters,
    )

    pt_model = ParticleTrackerV1(
        particle_detector_model=pd_model,
        particle_matcher_model=pm_model,
        cell_size=opt.cell_size,
        max_num_in_cell=opt.max_num_in_cell,
        basic_pow=opt.basic_pow,
        desc_pow=opt.desc_pow,
        match_threshold=opt.match_threshold,
    )
    if train:
        if opt.particle_tracker_checkpoint_path is None:
            _module_init(pt_model)
        else:
            pt_model.load_state_dict(torch.load(opt.particle_tracker_checkpoint_path, map_location='cpu'))

        if opt.particle_detector_checkpoint_path is None:
            print('\n---Train Detector & Matcher together!---\n')
        else:
            # 保存的pt_model 里包含 pd_model，因此最后载入 pd_model 使得超参数一致的情况下 pd_model 可替换
            pd_model.load_state_dict(torch.load(opt.particle_detector_checkpoint_path, map_location='cpu'))
            pd_model.requires_grad_(requires_grad=False)  # 锁定 particle_detector 中的参数
    else:
        if opt.particle_tracker_checkpoint_path is None:
            raise ValueError("Should have particle_tracker_checkpoint_path!")
        else:
            pt_model.load_state_dict(torch.load(opt.particle_tracker_checkpoint_path, map_location='cpu'))

    if not opt.force_cpu:
        device = torch.device(f'cuda:{opt.cuda_ids[0]}' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    pt_model.to(device)
    if 'cuda' in str(device):
        pt_model = torch.nn.DataParallel(pt_model, device_ids=opt.cuda_ids)
        print(f'{mode} on: {device}, with device_ids: {opt.cuda_ids}')
    else:
        print(f'{mode} on: {device}')

    return pt_model, device
