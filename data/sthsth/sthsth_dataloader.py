"""Something-Something V2 DataLoader with train/val/test splits."""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.sthsth.sthsth_dataset import STHSTHDataset
from utils.essential import get_cfg


def collate_fn(batch):
    """Custom collate: stack tensors, convert labels to long tensor."""
    frames, labels, prompts, prompt_inputs, prompt_masks = zip(*batch)
    return (
        torch.stack(frames),                         # (B, C, T, H, W)
        torch.tensor(labels, dtype=torch.long),      # (B,)
        prompts,                                      # tuple of str
        torch.stack(prompt_inputs),                   # (B, max_length)
        torch.stack(prompt_masks),                    # (B, max_length)
    )


def sthsth_dataloader(cfg):
    """Build Something-Something V2 train/val/test DataLoaders from config."""
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])

    common_kwargs = dict(
        video_dir=cfg.data.dir.video_dir,
        class_label_mapping_file=cfg.data.dir.class_label_mapping_file,
        frames_per_clip=cfg.data.setting.num_frames,
        transform=transform,
        tokenizer=cfg.data.setting.tokenizer,
    )

    train_dataset = STHSTHDataset(annotation_file=cfg.data.dir.train_anno, **common_kwargs)
    valid_dataset = STHSTHDataset(annotation_file=cfg.data.dir.val_anno, **common_kwargs)
    test_dataset = STHSTHDataset(annotation_file=cfg.data.dir.test_anno, **common_kwargs)

    loader_kwargs = dict(
        batch_size=cfg.data.setting.batch_size,
        num_workers=cfg.data.setting.num_workers,
        collate_fn=collate_fn,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, valid_loader, test_loader
