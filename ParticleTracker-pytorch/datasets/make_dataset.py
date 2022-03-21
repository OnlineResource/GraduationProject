from torch.utils.data import DataLoader

from datasets.particle_dataset import ParticleMatcherDataset, ParticleDetectorDataset


def pd_dataset_preparation(opt):
    train_set = ParticleDetectorDataset(
        file_path=opt.datasets_path,
        purpose='train',
        cell_size=opt.cell_size,
    )
    train_loader = DataLoader(
        dataset=train_set,
        shuffle=True,
        batch_size=opt.batch_size,
        drop_last=True,
    )

    valid_set = ParticleDetectorDataset(
        file_path=opt.datasets_path,
        purpose='valid',
        cell_size=opt.cell_size,
    )
    valid_loader = DataLoader(
        dataset=valid_set,
        shuffle=True,
        batch_size=opt.batch_size,
        drop_last=True,
    )
    return train_loader, valid_loader


def pm_dataset_preparation(opt):
    train_set = ParticleMatcherDataset(
        file_path=opt.datasets_path,
        purpose='train',
        cell_size=opt.cell_size,
    )
    train_loader = DataLoader(
        dataset=train_set,
        shuffle=True,
        batch_size=opt.batch_size,
        drop_last=True,
        num_workers=opt.num_workers,
    )

    valid_set = ParticleMatcherDataset(
        file_path=opt.datasets_path,
        purpose='valid',
        cell_size=opt.cell_size,
    )
    valid_loader = DataLoader(
        dataset=valid_set,
        shuffle=True,
        batch_size=opt.batch_size,
        drop_last=True,
        num_workers=opt.num_workers,
    )

    return train_loader, valid_loader
