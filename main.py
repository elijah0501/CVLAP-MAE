"""CVLAP / CVLAP-MAE training entry point."""

import os

# ── Environment variables (MUST be set before any torch / CUDA import) ──────
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import gc
import logging

import torch
import pytorch_lightning as pl

from utils.essential import get_cfg, ModelFactory, dir_creation

# Utilise Tensor-Core acceleration on Ampere+ GPUs
torch.set_float32_matmul_precision("high")

log = logging.getLogger(__name__)


"""
Experiment configs ordered by paper table appearance.

─── Table 1: Main results (CVLAP-MAE-B / CVLAP-MAE-L) ───────────────────────
Pre-training:
  config/pretraining/VideoAudioText_Kinetics_CVLAP_MAE_base.yaml
  config/pretraining/VideoAudioText_Kinetics_CVLAP_MAE_large.yaml
Fine-tuning:
  config/finetuning/VideoTextAudio_Kinetics_CVLAP_MAE_base_UCF101.yaml
  config/finetuning/VideoTextAudio_Kinetics_CVLAP_MAE_base_HMDB51.yaml
  config/finetuning/VideoTextAudio_Kinetics_CVLAP_MAE_base_SSv2.yaml
  config/finetuning/VideoTextAudio_Kinetics_CVLAP_MAE_large_UCF101.yaml
  config/finetuning/VideoTextAudio_Kinetics_CVLAP_MAE_large_HMDB51.yaml
  config/finetuning/VideoTextAudio_Kinetics_CVLAP_MAE_large_SSv2.yaml

─── Table 2: Full Kinetics-400 ────────────────────────────────────────────────
  (Same as Table 1 CVLAP-MAE-L configs, set subset_ratio=1.0)

─── Table 4: Zero-shot HMDB-51 ────────────────────────────────────────────────
  config/zeroshot/ZeroShot_CVLAP_MAE_base_HMDB51.yaml
  config/zeroshot/ZeroShot_CVLAP_MAE_large_HMDB51.yaml

─── Table 5: Audio ablation (with vs without audio) ───────────────────────────
Pre-training (without audio):
  config/pretraining/VideoText_Kinetics_CVLAP_MAE_large.yaml
Fine-tuning (without audio):
  config/finetuning/VideoText_Kinetics_CVLAP_MAE_large_UCF101.yaml
  config/finetuning/VideoText_Kinetics_CVLAP_MAE_large_SSv2.yaml

─── Table 6: MAE ablation (with vs without MAE) ───────────────────────────────
Pre-training (without MAE, contrastive only):
  config/pretraining/VideoAudioText_Kinetics_CVLAP_base.yaml
  config/pretraining/VideoAudioText_Kinetics_CVLAP_large.yaml
Fine-tuning (without MAE):
  config/finetuning/VideoAudioText_Kinetics_CVLAP_large_UCF101.yaml
  config/finetuning/VideoAudioText_Kinetics_CVLAP_large_SSv2.yaml

─── Table 7: Parameter sharing ablation ───────────────────────────────────────
Pre-training (shared):
  config/pretraining/VideoAudioText_Kinetics_CVLAP_MAE_large_shared.yaml
Fine-tuning (shared):
  config/finetuning/VideoTextAudio_Kinetics_CVLAP_MAE_large_shared_UCF101.yaml

─── Table 8: Masking ratio ablation ───────────────────────────────────────────
  (Same as Table 1 CVLAP-MAE-L configs, change mask_ratio to 0.50/0.75/0.90)

─── Table 9: Asymmetric encoder size ──────────────────────────────────────────
Pre-training:
  config/pretraining/Video_large_AudioText_Kinetics_CVLAP_MAE_base.yaml
  config/pretraining/Audio_large_VideoText_Kinetics_CVLAP_MAE_base.yaml
  config/pretraining/Text_large_AudioText_Kinetics_CVLAP_MAE_base.yaml
Fine-tuning:
  config/finetuning/Video_large_AudioText_Kinetics_CVLAP_MAE_base_UCF101.yaml
  config/finetuning/Video_large_AudioText_Kinetics_CVLAP_MAE_base_HMDB51.yaml
  config/finetuning/Audio_large_VideoText_Kinetics_CVLAP_MAE_base_UCF101.yaml
  config/finetuning/Audio_large_VideoText_Kinetics_CVLAP_MAE_base_HMDB51.yaml
  config/finetuning/Text_large_AudioText_Kinetics_CVLAP_MAE_base_UCF101.yaml
  config/finetuning/Text_large_AudioText_Kinetics_CVLAP_MAE_base_HMDB51.yaml
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CVLAP")
    parser.add_argument(
        "-C", "--config",
        type=str,
        default="config/pretraining/VideoAudioText_Kinetics_CVLAP_base.yaml",
        help="Path to the YAML experiment config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_cfg(args.config)
    experiment = cfg.Experiment

    # Reproducibility
    pl.seed_everything(experiment.reproducibility.seed, workers=True)

    # Checkpoint directory
    checkpoint_path = os.path.join(
        experiment.checkpoints, str(experiment.reproducibility.seed)
    )
    dir_creation(path=checkpoint_path)
    cfg.Experiment.checkpoints = checkpoint_path

    # Launch training
    task_name = experiment.training_task
    # ``script`` specifies the consolidated module/class to import.
    # Falls back to ``name`` for backward compatibility with old configs.
    class_name = getattr(experiment, "script", experiment.name)
    model_factory = ModelFactory()
    wandb_project = "CVLAP-MAE"

    try:
        model_factory.create_model(task_name, class_name, wandb_project=wandb_project, **cfg)
    finally:
        # ── Cleanup: release GPU memory regardless of success / failure ──
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            log.info("GPU memory released.")


if __name__ == "__main__":
    main()
