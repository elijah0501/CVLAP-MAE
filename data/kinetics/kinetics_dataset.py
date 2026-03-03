"""Kinetics-400 Dataset with video frames, audio spectrograms, and CLIP text encoding."""

import os
import random
import signal
import warnings

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from clip import clip
from moviepy.editor import VideoFileClip
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.essential import get_cfg

# Suppress moviepy/librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── Audio constants ─────────────────────────────────────────────────────────
_AUDIO_TIMEOUT_SEC = 30   # max seconds allowed for a single audio extraction


class _AudioLoadTimeout(Exception):
    """Raised when audio extraction exceeds the allowed time."""


def _timeout_handler(signum, frame):
    raise _AudioLoadTimeout("Audio extraction timed out")


class KineticsDataset(Dataset):
    """Kinetics-400 video dataset with audio spectrogram and CLIP text encoding.

    Videos without extractable audio are automatically skipped and replaced
    by another randomly sampled video that does have valid audio.

    Returns:
        frames: (C, T, H, W) float32 tensor
        label_index: int class label
        video_prompt: str
        encoded_video_prompt: (1, 77) CLIP token ids
        audio_spectrogram: (1024, 128) float32 tensor
        audio_prompt: str
        encoded_audio_prompt: (1, 77) CLIP token ids
    """

    def __init__(self, video_dir, anno_dir, label_mapping,
                 frames_per_clip=16, sr=44100, n_fft=2048, hop_length=512,
                 transform=None, tokenizer=None, subset_ratio=1.0):
        super().__init__()
        self.video_dir = video_dir
        self.annotations = pd.read_csv(anno_dir, header=0)
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.tokenizer = tokenizer

        # Optionally subsample the dataset (stratified by label)
        if 0.0 < subset_ratio < 1.0:
            self.annotations = (
                self.annotations
                .groupby('label', group_keys=False)
                .apply(lambda g: g.sample(frac=subset_ratio, random_state=42))
                .reset_index(drop=True)
            )

        self.class_labels_dict = self._load_class_labels(label_mapping)
        self.video_index = self.annotations['label'].map(self.class_labels_dict).tolist()

    def __len__(self):
        return len(self.annotations)

    _MAX_AUDIO_RETRIES = 10

    def __getitem__(self, idx):
        for _attempt in range(self._MAX_AUDIO_RETRIES):
            row = self.annotations.iloc[idx]
            video_path = os.path.join(
                self.video_dir,
                f"{row['youtube_id']}_{row['time_start']:06d}_{row['time_end']:06d}.mp4"
            )
            label_index = self.class_labels_dict[row['label']]
            text_label = row['label']

            # Text prompts
            video_prompt = f"This is a video of {text_label}."
            encoded_video_prompt = clip.tokenize(video_prompt)

            # Video frames
            frames = self._load_video_frames(video_path)
            if self.transform:
                frames = [self.transform(frame) for frame in frames]
            frames = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)

            # Audio — skip this video if audio is unavailable
            audio_result = self._load_audio_data(video_path, text_label)
            if audio_result is not None:
                audio_spectrogram, audio_prompt, encoded_audio_prompt = audio_result
                return (frames, label_index, video_prompt, encoded_video_prompt,
                        audio_spectrogram, audio_prompt, encoded_audio_prompt)

            # Pick another random sample and retry
            idx = random.randint(0, len(self) - 1)

        raise RuntimeError(
            f"Failed to find a sample with valid audio after "
            f"{self._MAX_AUDIO_RETRIES} retries (last video: {video_path})."
        )

    @staticmethod
    def _load_class_labels(class_label_file):
        """Load CSV label mapping: {text_label: numeric_label}."""
        df = pd.read_csv(class_label_file)
        return {row.iloc[1]: int(row.iloc[0]) for _, row in df.iterrows()}

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
            frames.append(random.choice(frames))
        return frames[:self.frames_per_clip]

    def _load_audio_data(self, video_path, text_label):
        """Extract audio spectrogram from video, resized to (1024, 128).

        Returns a (spectrogram, prompt, encoded_prompt) tuple on success,
        or ``None`` when the video has no audio track or audio extraction
        fails for any reason.  The caller is expected to discard the
        sample and retry with a different video.

        Uses a SIGALRM-based timeout so that corrupted / extremely long
        videos cannot block a DataLoader worker indefinitely (which would
        trigger an NCCL collective timeout under DDP).
        """
        # Set up a timeout alarm (only works in the main thread of each
        # DataLoader worker on POSIX systems).
        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(_AUDIO_TIMEOUT_SEC)
        except (ValueError, OSError):
            # signal.signal can only be called from the main thread;
            # if this is a non-main thread, skip the alarm.
            old_handler = None

        try:
            video_clip = VideoFileClip(video_path)
            try:
                audio = video_clip.audio
                if audio is None:
                    print(f"No audio track in {video_path}, skipping.")
                    return None

                audio_array = audio.to_soundarray(fps=self.sr).mean(axis=1)
                s = librosa.stft(audio_array, n_fft=self.n_fft, hop_length=self.hop_length)
                s_db = librosa.amplitude_to_db(np.abs(s))
                s_db = torch.tensor(s_db, dtype=torch.float32)

                # Resize to fixed (1024, 128)
                s_db = F.interpolate(
                    s_db.unsqueeze(0).unsqueeze(0),
                    size=(1024, 128), mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0)

                audio_prompt = f"This is the sound of {text_label}."
                return s_db, audio_prompt, clip.tokenize(audio_prompt)
            finally:
                video_clip.close()

        except _AudioLoadTimeout:
            print(f"Audio load TIMEOUT ({_AUDIO_TIMEOUT_SEC}s) for {video_path}, skipping.")
            return None
        except (IndexError, UnicodeDecodeError, OSError) as e:
            print(f"Audio load error for {video_path}: {e}, skipping.")
            return None
        finally:
            # Cancel the alarm and restore the old handler.
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (ValueError, OSError):
                pass
