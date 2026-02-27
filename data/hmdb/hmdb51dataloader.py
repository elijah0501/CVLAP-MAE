"""HMDB51 DataLoader with train/val/test split."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from data.hmdb.hmdb51dataset import HMDB51Dataset
from utils.essential import get_cfg


def split_dataset(dataset, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train/val/test by ratio."""
    n = len(dataset)
    val_size = int(val_ratio * n)
    test_size = int(test_ratio * n)
    train_size = n - val_size - test_size
    return random_split(dataset, [train_size, val_size, test_size])


def collate_fn(batch):
    """Custom collate: stack tensors, convert labels to long tensor."""
    frames, labels, _video_prompts, prompt_ids, prompt_masks = zip(*batch)
    return (
        torch.stack(frames),                    # (B, C, T, H, W)
        torch.tensor(labels, dtype=torch.long), # (B,)
        torch.stack(prompt_ids),                # (B, max_length)
        torch.stack(prompt_masks),              # (B, max_length)
    )


def hmdb51_dataloader(cfg):
    """Build HMDB51 train/val/test DataLoaders from config."""
    transform = transforms.Compose([
        transforms.Resize((cfg.plain_model.video.input_size, cfg.plain_model.video.input_size)),
        transforms.ToTensor(),
    ])

    dataset = HMDB51Dataset(
        root_dir=cfg.data.dir.root_dir,
        frames_per_clip=cfg.data.setting.num_frames,
        transform=transform,
        tokenizer=cfg.data.setting.tokenizer,
    )

    train_set, val_set, test_set = split_dataset(dataset, val_ratio=0.1, test_ratio=0.1)

    loader_kwargs = dict(
        batch_size=cfg.data.setting.batch_size,
        num_workers=cfg.data.setting.num_workers,
        collate_fn=collate_fn,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
