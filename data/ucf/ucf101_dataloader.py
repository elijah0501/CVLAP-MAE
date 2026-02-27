"""UCF-101 DataLoader (HuggingFace tokenizer variant) with train/val/test splits."""

import os
import pandas as pd
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from data.ucf.ucf101_dataset import UCF101Dataset
from utils.essential import get_cfg


def split_dataset(annotation_file, test_size=0.2, random_state=42):
    """Split annotation file into train/val DataFrames."""
    annotations = pd.read_csv(annotation_file, header=None, delimiter=' ')
    train_ann, val_ann = train_test_split(annotations, test_size=test_size, random_state=random_state)
    return train_ann, val_ann


def collate_fn(batch):
    """Custom collate: stack all tensors, convert labels to long tensor."""
    (frames, labels, video_prompts, encoded_vps, encoded_vp_masks,
     audio_specs, audio_prompts, encoded_aps, encoded_ap_masks) = zip(*batch)

    return (
        torch.stack(frames),                    # (B, C, T, H, W)
        torch.tensor(labels, dtype=torch.long), # (B,)
        video_prompts,                           # tuple of str
        torch.stack(encoded_vps),               # (B, 77)
        torch.stack(encoded_vp_masks),          # (B, 77)
        torch.stack(audio_specs),               # (B, 1024, 128)
        audio_prompts,                           # tuple of str
        torch.stack(encoded_aps),               # (B, 77)
        torch.stack(encoded_ap_masks),          # (B, 77)
    )


def ucf101_dataloader(cfg):
    """Build UCF-101 train/val/test DataLoaders from config."""
    train_ann, val_ann = split_dataset(cfg.data.dir.train_anno, test_size=0.2)

    transform = transforms.Compose([
        transforms.Resize((cfg.plain_model.video.input_size, cfg.plain_model.video.input_size)),
        transforms.ToTensor(),
    ])

    common_kwargs = dict(
        root_dir=cfg.data.dir.root_dir,
        class_label_file=cfg.data.dir.class_label_file,
        frames_per_clip=cfg.data.setting.num_frames,
        sr=cfg.data.setting.sr,
        n_fft=cfg.data.setting.n_fft,
        hop_length=cfg.data.setting.hop_length,
        transform=transform,
        tokenizer=cfg.data.setting.tokenizer,
    )

    train_dataset = UCF101Dataset(annotation_file=train_ann, **common_kwargs)
    valid_dataset = UCF101Dataset(annotation_file=val_ann, **common_kwargs)
    test_dataset = UCF101Dataset(annotation_file=cfg.data.dir.test_anno, **common_kwargs)

    loader_kwargs = dict(
        batch_size=cfg.data.setting.batch_size,
        num_workers=cfg.data.setting.num_workers,
        collate_fn=collate_fn,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, valid_loader, test_loader

