"""HMDB51 Dataset for video classification with text prompts."""

import os

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

from utils.essential import get_cfg


class HMDB51Dataset(Dataset):
    """HMDB51 video dataset with text prompt generation and tokenization.

    Each sample returns video frames, class label, text prompt,
    and tokenized prompt (input_ids + attention_mask).
    """

    def __init__(self, root_dir, frames_per_clip=16, transform=None, tokenizer='google/mt5-base'):
        super().__init__()
        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, legacy=False)

        self.video_files, self.labels = self._load_videos_and_labels()
        self.class_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}

    def _load_videos_and_labels(self):
        """Scan root directory for .avi files organized by class folders."""
        video_files, labels = [], []
        for label in sorted(os.listdir(self.root_dir)):
            label_dir = os.path.join(self.root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for root, _, files in os.walk(label_dir):
                for f in sorted(files):
                    if f.endswith('.avi'):
                        video_files.append(os.path.join(root, f))
                        labels.append(label)
        return video_files, labels

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.class_to_idx[self.labels[idx]]
        text_label = self.labels[idx]
        video_prompt = f"This is a video of {text_label}."
        prompt_ids, prompt_mask = self._tokenize(video_prompt)

        frames = self._load_video_frames(video_path)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # (T, C, H, W) -> (C, T, H, W)
        frames = torch.stack(frames).permute(1, 0, 2, 3)
        return frames, label, video_prompt, prompt_ids, prompt_mask

    def _tokenize(self, text):
        """Tokenize text with padding and truncation."""
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

    def _load_video_frames(self, video_path):
        """Load uniformly sampled frames from video using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                return [Image.new('RGB', (224, 224))] * self.frames_per_clip

            # Uniformly sample frame indices
            indices = torch.linspace(0, frame_count - 1, self.frames_per_clip).long().tolist()
            frames = []
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            cap.release()

        # Pad with last frame if needed
        if not frames:
            return [Image.new('RGB', (224, 224))] * self.frames_per_clip
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1])
        return frames[:self.frames_per_clip]
