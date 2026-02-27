"""UCF-101 Dataset with video frames, audio spectrograms, and HuggingFace text encoding."""

import os
import re
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from moviepy.editor import VideoFileClip
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

from utils.essential import get_cfg

warnings.filterwarnings("ignore", category=UserWarning)


class UCF101Dataset(Dataset):
    """UCF-101 video dataset with audio spectrogram and HuggingFace tokenizer.

    Returns:
        frames: (C, T, H, W) float32 tensor
        label: int (0-indexed)
        video_prompt: str
        encoded_video_prompt: (max_length,) token ids
        encoded_video_prompt_mask: (max_length,) attention mask
        audio_spectrogram: (1024, 128) float32 tensor (-1 if no audio)
        audio_prompt: str
        encoded_audio_prompt: (max_length,) token ids
        encoded_audio_prompt_mask: (max_length,) attention mask
    """

    def __init__(self, root_dir, annotation_file, class_label_file,
                 frames_per_clip=16, sr=44100, n_fft=2048, hop_length=512,
                 transform=None, tokenizer='google/mt5-base'):
        super().__init__()
        self.root_dir = root_dir
        self.annotations = (annotation_file if isinstance(annotation_file, pd.DataFrame)
                            else pd.read_csv(annotation_file, header=None, delimiter=' '))
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, legacy=False)

        self.video_files = self.annotations[0].values
        self.labels = self.annotations[1].values if len(self.annotations.columns) > 1 else None
        self.class_labels = self._load_class_labels(class_label_file)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.video_files[idx])

        if self.labels is None:
            raise ValueError(f"Video file {video_path} has no labels")

        label = self.labels[idx] - 1  # 0-indexed
        if label < 0 or label >= len(self.class_labels):
            raise ValueError(f"Label {label} out of bounds for index {idx}")

        text_label = self._process_label(self.class_labels[label + 1])
        video_prompt = f"This is a video of {text_label}."
        encoded_vp, encoded_vp_mask = self._tokenize(video_prompt)

        frames = self._load_video_frames(video_path)
        audio_spec, audio_prompt, encoded_ap, encoded_ap_mask = self._load_audio_data(video_path, text_label)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)

        return (frames, label, video_prompt, encoded_vp, encoded_vp_mask,
                audio_spec, audio_prompt, encoded_ap, encoded_ap_mask)

    @staticmethod
    def _load_class_labels(class_label_file):
        """Load class labels from text file: {numeric_label: text_label}."""
        class_labels = {}
        with open(class_label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    class_labels[int(parts[0])] = parts[1]
        return class_labels

    @staticmethod
    def _process_label(label):
        """Convert CamelCase label to space-separated lowercase."""
        return re.sub(r'(?<!^)(?=[A-Z])', ' ', label).lower()

    def _tokenize(self, text):
        """Tokenize with padding and truncation, return ids + mask."""
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

    def _load_video_frames(self, video_path):
        """Load uniformly sampled frames from video."""
        cap = cv2.VideoCapture(video_path)
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                return [Image.new('RGB', (224, 224))] * self.frames_per_clip

            indices = torch.linspace(0, frame_count - 1, self.frames_per_clip).long().tolist()
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

    def _load_audio_data(self, video_path, text_label):
        """Extract audio spectrogram, resized to (1024, 128)."""
        no_audio_prompt = "No Audio."
        no_audio_ids, no_audio_mask = self._tokenize(no_audio_prompt)
        no_audio = (torch.full((1024, 128), -1.0), no_audio_prompt, no_audio_ids, no_audio_mask)

        try:
            video_clip = VideoFileClip(video_path)
            try:
                audio = video_clip.audio
                if audio is None:
                    return no_audio

                audio_array = audio.to_soundarray(fps=self.sr).mean(axis=1)
                s = librosa.stft(audio_array, n_fft=self.n_fft, hop_length=self.hop_length)
                s_db = librosa.amplitude_to_db(np.abs(s))
                s_db = torch.tensor(s_db, dtype=torch.float32)

                s_db = F.interpolate(
                    s_db.unsqueeze(0).unsqueeze(0),
                    size=(1024, 128), mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0)

                audio_prompt = f"This is an audio of {text_label}."
                encoded_ap, encoded_ap_mask = self._tokenize(audio_prompt)
                return s_db, audio_prompt, encoded_ap, encoded_ap_mask
            finally:
                video_clip.close()

        except (UnicodeDecodeError, OSError) as e:
            print(f"Audio load error for {video_path}: {e}")
            return no_audio
