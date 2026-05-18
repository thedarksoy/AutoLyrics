"""
AUTOLYRICS - Script 1: Baseline Data Preparation & Zero-Shot Evaluation
========================================================================
Loads the DALI-style singing dataset, preprocesses audio (16kHz, 30s chunks),
normalizes text, creates train/test splits, and runs zero-shot Whisper-small
inference to establish baseline WER/CER metrics.

Hardware target: RTX 4060 (8GB VRAM)
"""

import os
import re
import json
import time
import gc
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, Audio, DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Model
    "model_name": "openai/whisper-small",
    "language": "en",
    "task": "transcribe",

    # Audio
    "target_sample_rate": 16000,
    "max_duration_seconds": 30,

    # Dataset
    # Option A: HuggingFace-hosted singing dataset (recommended for easy setup)
    "hf_dataset_name": "DynamicSuperb/SongLyricRecognition_SingSet",
    "hf_dataset_split": "test",  # SingSet only has 'test'
    "hf_audio_column": "audio",
    "hf_text_column": "label",

    # Option B: Local DALI data (set use_local_dali=True and provide paths)
    "use_local_dali": False,
    "local_audio_dir": "data/dali/audio",
    "local_annotations_dir": "data/dali/annotations",

    # Splits
    "test_size": 0.2,
    "seed": 42,

    # Output
    "output_dir": "outputs",
    "processed_data_dir": "data/processed",
    "baseline_results_file": "outputs/baseline_results.json",
}


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================
def normalize_text(text: str) -> str:
    """
    Normalize ground-truth lyrics for fair WER/CER comparison.
    - Lowercase
    - Strip punctuation
    - Remove annotations like [Chorus], (Verse 1), etc.
    - Collapse whitespace
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove bracketed annotations
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    # Remove punctuation (keep apostrophes for contractions)
    text = re.sub(r"[^\w\s']", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================================
# AUDIO PREPROCESSING
# ============================================================================
def preprocess_audio_array(audio_array: np.ndarray, source_sr: int, target_sr: int, max_samples: int) -> np.ndarray:
    """
    Resample audio to target_sr and truncate to max_samples.
    Returns a numpy array.
    """
    if source_sr != target_sr:
        waveform = torch.tensor(audio_array, dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=target_sr)
        waveform = resampler(waveform)
        audio_array = waveform.squeeze(0).numpy()

    # Truncate to max duration
    if len(audio_array) > max_samples:
        audio_array = audio_array[:max_samples]

    return audio_array


# ============================================================================
# DATASET LOADING
# ============================================================================
def load_hf_singing_dataset(config: dict) -> DatasetDict:
    """
    Load a singing lyrics dataset from Hugging Face and create train/test splits.
    Uses datasets 3.x which decodes audio via soundfile (no torchcodec needed).
    """
    print(f"[DATA] Loading HF dataset: {config['hf_dataset_name']}")
    ds = load_dataset(config["hf_dataset_name"])

    # The SingSet dataset may only have one split
    if isinstance(ds, DatasetDict):
        available_splits = list(ds.keys())
        print(f"[DATA] Available splits: {available_splits}")
        if "train" in ds and "test" in ds:
            # Cast audio to target sample rate
            for split_name in ds:
                ds[split_name] = ds[split_name].cast_column(
                    config["hf_audio_column"], Audio(sampling_rate=config["target_sample_rate"])
                )
            return ds
        raw = ds[available_splits[0]]
    else:
        raw = ds

    print(f"[DATA] Raw dataset size: {len(raw)}")

    # Cast audio column to auto-decode + resample to 16kHz
    if config["hf_audio_column"] in raw.column_names:
        raw = raw.cast_column(config["hf_audio_column"], Audio(sampling_rate=config["target_sample_rate"]))

    # Create train/test split
    split = raw.train_test_split(test_size=config["test_size"], seed=config["seed"])
    print(f"[DATA] Train: {len(split['train'])}, Test: {len(split['test'])}")
    return split


def load_local_dali_dataset(config: dict) -> DatasetDict:
    """
    Load DALI dataset from local files using audiofolder format.
    Expects: audio files in audio_dir, a metadata.csv mapping file_name -> lyrics.
    """
    audio_dir = Path(config["local_audio_dir"])
    if not audio_dir.exists():
        raise FileNotFoundError(
            f"DALI audio directory not found: {audio_dir}\n"
            "Please download the DALI dataset from https://zenodo.org and place audio files here.\n"
            "See: https://github.com/gabolsgabs/DALI for instructions."
        )

    print(f"[DATA] Loading local DALI data from: {audio_dir}")
    ds = load_dataset("audiofolder", data_dir=str(audio_dir))

    if isinstance(ds, DatasetDict) and "train" in ds:
        raw = ds["train"]
    else:
        raw = ds

    raw = raw.cast_column("audio", Audio(sampling_rate=config["target_sample_rate"]))
    split = raw.train_test_split(test_size=config["test_size"], seed=config["seed"])
    print(f"[DATA] Train: {len(split['train'])}, Test: {len(split['test'])}")
    return split


def load_dataset_pipeline(config: dict) -> DatasetDict:
    """Main entry point: pick local DALI or HF dataset."""
    if config["use_local_dali"]:
        return load_local_dali_dataset(config)
    else:
        return load_hf_singing_dataset(config)


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================
def preprocess_dataset(dataset_dict: DatasetDict, config: dict) -> DatasetDict:
    """
    Apply audio truncation (max 30s) and text normalization to the dataset.
    Audio is already decoded and resampled to 16kHz by the Audio() cast.
    """
    max_samples = config["target_sample_rate"] * config["max_duration_seconds"]
    audio_col = config["hf_audio_column"]
    text_col = config["hf_text_column"]

    def process_example(example):
        # Audio: already decoded + resampled by datasets Audio() feature
        audio = example[audio_col]
        audio_array = np.array(audio["array"], dtype=np.float32)

        # Truncate to max duration
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]

        example[audio_col] = {
            "array": audio_array,
            "sampling_rate": config["target_sample_rate"],
            "path": audio.get("path", ""),
        }

        # Text normalization
        raw_text = example.get(text_col, "")
        example["normalized_text"] = normalize_text(raw_text)

        return example

    print("[PREPROCESS] Processing dataset...")
    for split_name in dataset_dict:
        print(f"  Processing split: {split_name} ({len(dataset_dict[split_name])} examples)")
        dataset_dict[split_name] = dataset_dict[split_name].map(
            process_example,
            desc=f"Preprocessing {split_name}",
        )

    # Filter out examples with empty text
    for split_name in dataset_dict:
        before = len(dataset_dict[split_name])
        dataset_dict[split_name] = dataset_dict[split_name].filter(
            lambda x: len(x["normalized_text"].strip()) > 0
        )
        after = len(dataset_dict[split_name])
        if before != after:
            print(f"  [{split_name}] Filtered {before - after} examples with empty text")

    return dataset_dict


# ============================================================================
# ZERO-SHOT BASELINE EVALUATION
# ============================================================================
def run_baseline_evaluation(dataset_dict: DatasetDict, config: dict) -> dict:
    """
    Run zero-shot inference with whisper-small on the test split.
    Returns baseline WER, CER, latency, and VRAM stats.
    """
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION (Zero-Shot Whisper-Small)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BASELINE] Device: {device}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"[BASELINE] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[BASELINE] VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model and processor
    print(f"[BASELINE] Loading model: {config['model_name']}")
    processor = WhisperProcessor.from_pretrained(config["model_name"])
    model = WhisperForConditionalGeneration.from_pretrained(config["model_name"]).to(device)
    model.eval()

    if device == "cuda":
        vram_after_load = torch.cuda.max_memory_allocated() / 1e9
        print(f"[BASELINE] VRAM after model load: {vram_after_load:.2f} GB")

    # Use language/task directly in generate() (modern API, avoids
    # deprecated forced_decoder_ids which conflict with max_target_positions)

    test_ds = dataset_dict["test"]
    audio_col = config["hf_audio_column"]

    all_references = []
    all_predictions = []
    total_inference_time = 0.0
    total_audio_seconds = 0.0

    print(f"[BASELINE] Running inference on {len(test_ds)} test examples...")

    for i in tqdm(range(len(test_ds)), desc="Baseline Inference"):
        example = test_ds[i]
        audio = example[audio_col]
        audio_array = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        reference = example["normalized_text"]

        if not reference.strip():
            continue

        # Compute audio duration
        audio_duration = len(audio_array) / sr
        total_audio_seconds += audio_duration

        # Prepare input features
        input_features = processor(
            audio_array, sampling_rate=sr, return_tensors="pt"
        ).input_features.to(device)

        # Inference with timing
        start_time = time.time()
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language=config["language"],
                task=config["task"],
                max_new_tokens=444,
            )
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        # Decode prediction
        prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        prediction = normalize_text(prediction)

        all_references.append(reference)
        all_predictions.append(prediction)

        # Print first few examples
        if i < 3:
            print(f"\n  Example {i}:")
            print(f"    REF: {reference[:100]}")
            print(f"    HYP: {prediction[:100]}")
            print(f"    Time: {inference_time:.2f}s | Audio: {audio_duration:.1f}s")

    # Calculate metrics
    if not all_references:
        print("[BASELINE] ERROR: No valid examples found for evaluation!")
        return {}

    baseline_wer = wer(all_references, all_predictions)
    baseline_cer = cer(all_references, all_predictions)

    # VRAM stats
    peak_vram = 0.0
    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / 1e9

    # Latency stats
    avg_latency = total_inference_time / len(all_references) if all_references else 0
    rtf = total_inference_time / total_audio_seconds if total_audio_seconds > 0 else 0

    results = {
        "model": config["model_name"],
        "approach": "zero-shot (baseline)",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
        "num_test_examples": len(all_references),
        "wer": round(baseline_wer, 4),
        "cer": round(baseline_cer, 4),
        "avg_latency_seconds": round(avg_latency, 3),
        "real_time_factor": round(rtf, 3),
        "peak_vram_gb": round(peak_vram, 2),
        "total_audio_seconds": round(total_audio_seconds, 1),
        "total_inference_seconds": round(total_inference_time, 1),
    }

    print("\n" + "-" * 40)
    print("BASELINE RESULTS")
    print("-" * 40)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("-" * 40)

    # Cleanup
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


# ============================================================================
# SAVE UTILITIES
# ============================================================================
def save_results(results: dict, filepath: str):
    """Save evaluation results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE] Results saved to: {filepath}")


def save_processed_dataset(dataset_dict: DatasetDict, output_dir: str):
    """Save the processed dataset to disk for reuse in training."""
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    print(f"[SAVE] Processed dataset saved to: {output_dir}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("AUTOLYRICS - Phase 1 & 2: Data Preprocessing + Baseline Eval")
    print("=" * 60)

    # Check CUDA availability
    print(f"\n[SYSTEM] PyTorch version: {torch.__version__}")
    print(f"[SYSTEM] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[SYSTEM] CUDA version: {torch.version.cuda}")
        print(f"[SYSTEM] GPU: {torch.cuda.get_device_name(0)}")

    # Phase 1: Load and preprocess dataset
    print("\n--- PHASE 1: Data Loading & Preprocessing ---")
    dataset_dict = load_dataset_pipeline(CONFIG)
    dataset_dict = preprocess_dataset(dataset_dict, CONFIG)

    # Save processed data
    save_processed_dataset(dataset_dict, CONFIG["processed_data_dir"])

    # Phase 2: Baseline evaluation
    print("\n--- PHASE 2: Zero-Shot Baseline Evaluation ---")
    baseline_results = run_baseline_evaluation(dataset_dict, CONFIG)

    if baseline_results:
        save_results(baseline_results, CONFIG["baseline_results_file"])

    print("\n[DONE] Phase 1 & 2 complete.")
    print(f"  Processed data: {CONFIG['processed_data_dir']}")
    print(f"  Baseline results: {CONFIG['baseline_results_file']}")
    print("\nNext step: Run 2_train_lora.py to fine-tune with LoRA.")


if __name__ == "__main__":
    main()
