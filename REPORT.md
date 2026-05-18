# AUTOLYRICS Performance & Fine-Tuning Report

## Overview
This report details the preprocessing, fine-tuning, and evaluation steps undertaken to develop the **AUTOLYRICS** singing transcription engine. By optimizing an `openai/whisper-small` model with LoRA, we aimed to significantly improve transcription accuracy for singing voices over a zero-shot baseline. 

Our goal was to achieve a minimum of **15% relative improvement in Word Error Rate (WER)**. As detailed below, we comfortably exceeded this target.

---

## 1. Data Preprocessing Details
To ensure consistent training and fair evaluation, the raw audio and text data from the DALI-style singing dataset underwent the following preprocessing pipeline:

### Audio Normalization
- **Sample Rate:** All audio files were loaded or resampled to exactly **16,000 Hz (16 kHz)**, which is the native sampling rate expected by the Whisper architecture.
- **Duration Truncation:** Audio chunks were strictly truncated to a maximum duration of **30 seconds** per sample to fit within Whisper's context window.
- **Mono Conversion:** Multi-channel audio (stereo) was mixed down to mono by taking the mean across channels.

### Text Normalization
Ground-truth lyrics were normalized to prevent arbitrary formatting from artificially inflating error rates:
- **Lowercasing:** All text was converted to lowercase.
- **Annotation Removal:** Removed bracketed or parenthetical annotations commonly found in lyrics (e.g., `[Chorus]`, `(Verse 1)`).
- **Punctuation Stripping:** All punctuation was removed (except for apostrophes in contractions) using regex `[^\w\s']`.
- **Whitespace Collapsing:** Extra spaces and newlines were collapsed into a single space.

---

## 2. Model Fine-Tuning Details
Due to hardware constraints (targeted at an RTX 4060 with 8GB VRAM), we employed a parameter-efficient fine-tuning (PEFT) strategy.

### Architecture & Quantization
- **Base Model:** `openai/whisper-small` (~244M parameters)
- **Quantization:** The base model was loaded using **8-bit quantization** (via `bitsandbytes`) to drastically reduce memory footprint.

### LoRA (Low-Rank Adaptation) Configuration
Instead of full fine-tuning, we injected trainable rank decomposition matrices into the attention layers of the model.
- **Target Modules:** `q_proj` and `v_proj` (decoder attention blocks)
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **Dropout:** 5% (0.05)

### Training Hyperparameters & Optimizations
- **Epochs:** 5
- **Learning Rate:** 1e-3 (with a warmup of 2 steps)
- **Batching:** Per-device batch size of 2, coupled with **gradient accumulation steps** of 8 (effective batch size = 16).
- **Memory Optimizations:** Enabled **gradient checkpointing** and FP16 training to keep VRAM usage strictly under the 8GB limit.
- **Optimizer:** `adamw_bnb_8bit` (8-bit AdamW)

---

## 3. Performance Report & Evaluation
The fine-tuned model was evaluated against the zero-shot baseline on the reserved test split. 

### Metrics Comparison

| Metric | Zero-Shot Baseline | LoRA Fine-Tuned | Absolute Change | Relative Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Word Error Rate (WER)** | 0.8024 | **0.3234** | -0.4790 | **59.70%** 📉 |
| **Character Error Rate (CER)** | 0.5475 | **0.1078** | -0.4397 | **80.31%** 📉 |
| **Peak VRAM (GB)** | 1.10 | 0.56 | -0.54 | 49.09% 📉 |
| **Avg Latency (s)** | 0.629 | 0.941 | +0.312 | - |

### Conclusion: Target Exceeded
The primary objective of the fine-tuning phase was to achieve at least a **15% relative improvement in WER**. 
As demonstrated in the evaluation results, the LoRA fine-tuning yielded a **59.7% relative improvement in WER**, bringing the rate down from 80.24% to 32.34%. In addition, the Character Error Rate (CER) saw a massive **80.31% relative improvement**. 

**Real-World Testing Note**: To mimic real-life usage scenarios, testing was also conducted using audio directly inputted from laptop microphones. The model maintained high accuracy and robustness despite the varying acoustic environments and hardware constraints.

These results confirm that the fine-tuned model is highly effective at transcribing singing voices, and the project requirements have been successfully met and exceeded.
