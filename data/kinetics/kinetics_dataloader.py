"""Kinetics-400 DataLoader with train/val/test splits."""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.kinetics.kinetics_dataset import KineticsDataset
from utils.essential import get_cfg


def collate_fn(batch):
    """Custom collate: stack tensors, concatenate CLIP tokens along batch dim."""
    (frames, labels, video_prompts, encoded_video_prompts,
     audio_spectrograms, audio_prompts, encoded_audio_prompts) = zip(*batch)

    return (
        torch.stack(frames),                         # (B, C, T, H, W)
        torch.tensor(labels, dtype=torch.long),      # (B,)
        video_prompts,                                # tuple of str
        torch.cat(encoded_video_prompts, dim=0),     # (B, 77)
        torch.stack(audio_spectrograms),             # (B, 1024, 128)
        audio_prompts,                                # tuple of str
        torch.cat(encoded_audio_prompts, dim=0),     # (B, 77)
    )


def kinetics_dataloader(cfg):
    """Build Kinetics-400 train/val/test DataLoaders from config."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    common_kwargs = dict(
        label_mapping=cfg.data.dir.label_mapping,
        frames_per_clip=cfg.data.setting.num_frames,
        sr=cfg.data.setting.sr,
        n_fft=cfg.data.setting.n_fft,
        hop_length=cfg.data.setting.hop_length,
        transform=transform,
        tokenizer=cfg.data.setting.tokenizer,
    )

    train_dataset = KineticsDataset(video_dir=cfg.data.dir.train_dir, anno_dir=cfg.data.dir.train_anno, **common_kwargs)
    valid_dataset = KineticsDataset(video_dir=cfg.data.dir.val_dir, anno_dir=cfg.data.dir.val_anno, **common_kwargs)
    test_dataset = KineticsDataset(video_dir=cfg.data.dir.test_dir, anno_dir=cfg.data.dir.test_anno, **common_kwargs)

    loader_kwargs = dict(
        batch_size=cfg.data.setting.batch_size,
        num_workers=cfg.data.setting.num_workers,
        collate_fn=collate_fn,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, valid_loader, test_loader
