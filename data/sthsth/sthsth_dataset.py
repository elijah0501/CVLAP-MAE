"""Something-Something V2 Dataset for video classification with text prompts."""

import json
import os
import random
import re

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

from utils.essential import get_cfg


class STHSTHDataset(Dataset):
    """Something-Something V2 video dataset with text prompt tokenization.

    Returns:
        frames: (C, T, H, W) float32 tensor
        label_index: int class label
        prompt: str description
        prompt_input: (max_length,) token ids
        prompt_attention_mask: (max_length,) attention mask
    """

    def __init__(self, video_dir, annotation_file, class_label_mapping_file,
                 frames_per_clip=16, transform=None, tokenizer='google/mt5-base'):
        super().__init__()
        self.video_dir = video_dir
        self.frames_per_clip = frames_per_clip
        self.transform = transform

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        with open(class_label_mapping_file, 'r') as f:
            self.label_mapping = {k.strip(): int(v) for k, v in json.load(f).items()}

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, legacy=False)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        video_id = annotation['id']
        label_template = annotation['template'].strip()
        # Remove all bracket placeholders: [something] -> something, [part] -> part, etc.
        label_cleaned = re.sub(r'\[([^\]]+)\]', r'\1', label_template).lower()
        # Normalize to match label mapping keys (which are lowercase without brackets)
        label_index = None
        for key, val in self.label_mapping.items():
            if key.lower() == label_cleaned:
                label_index = val
                break
        if label_index is None:
            raise ValueError(f"Template '{label_template}' (cleaned: '{label_cleaned}') not found in label mapping.")

        prompt = annotation['label']

        # Load video frames via OpenCV (.webm supported)
        video_path = os.path.join(self.video_dir, f"{video_id}.webm")
        frames = self._load_video_frames(video_path)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # (T, C, H, W) -> (C, T, H, W) for consistency with other datasets
        frames = torch.stack(frames).permute(1, 0, 2, 3)

        # Tokenize prompt
        tokenized = self.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt',
        )
        prompt_input = tokenized.input_ids.squeeze(0)
        prompt_attention_mask = tokenized.attention_mask.squeeze(0)

        return frames, label_index, prompt, prompt_input, prompt_attention_mask

    def _load_video_frames(self, video_path):
        """Load uniformly sampled frames from video using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                return [Image.new('RGB', (224, 224))] * self.frames_per_clip

            if frame_count >= self.frames_per_clip:
                indices = sorted(random.sample(range(frame_count), self.frames_per_clip))
            else:
                indices = sorted(random.choices(range(frame_count), k=self.frames_per_clip))

            frames = []
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            cap.release()

        if not frames:
            return [Image.new('RGB', (224, 224))] * self.frames_per_clip
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1])
        return frames[:self.frames_per_clip]

