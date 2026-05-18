"""
AUTOLYRICS - Script 2: LoRA Fine-Tuning of Whisper-Small
=========================================================
Loads the preprocessed DALI singing dataset, quantizes whisper-small to 8-bit,
applies LoRA to decoder attention blocks, and trains with Seq2SeqTrainer.

Hardware target: RTX 4060 (8GB VRAM)
Memory strategy: 8-bit quantization + gradient accumulation + gradient checkpointing
"""

import os
import json
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import numpy as np
from datasets import load_from_disk, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Model
    "model_name": "openai/whisper-small",
    "language": "en",
    "task": "transcribe",

    # Data
    "processed_data_dir": "data/processed",
    "audio_column": "audio",
    "text_column": "normalized_text",
    "target_sample_rate": 16000,

    # LoRA config
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj",
        "v_proj",
    ],

    # Training - optimized for RTX 4060 (8GB VRAM)
    "output_dir": "outputs/whisper-small-lora",
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,  # effective batch = 2 * 8 = 16
    "num_train_epochs": 5,
    "learning_rate": 1e-3,
    "warmup_steps": 2,
    "logging_steps": 1,
    "eval_steps": 5,
    "save_steps": 5,
    "save_total_limit": 3,
    "fp16": True,
    "gradient_checkpointing": True,
    "dataloader_num_workers": 2,
    "remove_unused_columns": False,
    "label_names": ["labels"],
    "load_best_model_at_end": True,
    "metric_for_best_model": "wer",
    "greater_is_better": False,
    "report_to": "none",
}


# ============================================================================
# DATA COLLATOR
# ============================================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom data collator for Whisper fine-tuning.
    Handles padding of both input features and labels.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract input features and pad
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Extract labels and pad
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding token id with -100 so it's ignored by loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if it was appended during encoding
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ============================================================================
# DATASET PREPARATION
# ============================================================================
def prepare_dataset_for_training(dataset_dict: DatasetDict, processor: WhisperProcessor, config: dict) -> DatasetDict:
    """
    Transform the preprocessed dataset into the format expected by Whisper training.
    - Extract log-mel spectrogram features
    - Tokenize target text into label IDs
    """
    audio_col = config["audio_column"]
    text_col = config["text_column"]

    def prepare_example(example):
        audio = example[audio_col]
        audio_array = np.array(audio["array"], dtype=np.float32)

        # Extract mel spectrogram features
        input_features = processor.feature_extractor(
            audio_array,
            sampling_rate=config["target_sample_rate"],
            return_tensors="np",
        ).input_features[0]

        # Tokenize the target text
        labels = processor.tokenizer(example[text_col]).input_ids

        example["input_features"] = input_features
        example["labels"] = labels
        return example

    print("[TRAIN-DATA] Preparing features for training...")
    for split_name in dataset_dict:
        print(f"  Processing split: {split_name}")
        dataset_dict[split_name] = dataset_dict[split_name].map(
            prepare_example,
            remove_columns=dataset_dict[split_name].column_names,
            desc=f"Featurizing {split_name}",
        )

    return dataset_dict


# ============================================================================
# MODEL SETUP WITH QUANTIZATION + LoRA
# ============================================================================
def setup_quantized_lora_model(config: dict):
    """
    Load Whisper-small in 8-bit quantization and apply LoRA to decoder attention.
    Returns the PEFT model ready for training.
    """
    print(f"[MODEL] Loading {config['model_name']} with 8-bit quantization...")

    # 8-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    # Load quantized model
    model = WhisperForConditionalGeneration.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Prepare model for k-bit training (freezes non-LoRA params, handles layernorm)
    model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing for memory savings
    if config["gradient_checkpointing"]:
        model.config.use_cache = False  # Required for gradient checkpointing

    # Configure LoRA - targeting decoder attention blocks only
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    if torch.cuda.is_available():
        vram_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"[MODEL] VRAM after model + LoRA setup: {vram_used:.2f} GB")

    return model


# ============================================================================
# METRICS
# ============================================================================
def compute_metrics(pred, processor):
    """Compute WER metric during evaluation."""
    from jiwer import wer as compute_wer
    import re

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Normalize for fair comparison
    def normalize(text):
        text = text.lower()
        text = re.sub(r"[^\w\s']", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    pred_str = [normalize(p) for p in pred_str]
    label_str = [normalize(l) for l in label_str]

    # Filter empty references
    pairs = [(r, p) for r, p in zip(label_str, pred_str) if r.strip()]
    if not pairs:
        return {"wer": 1.0}

    refs, hyps = zip(*pairs)
    wer_score = compute_wer(list(refs), list(hyps))

    return {"wer": round(wer_score, 4)}


# ============================================================================
# TRAINING
# ============================================================================
def train(config: dict):
    """Main training function."""
    print("=" * 60)
    print("AUTOLYRICS - Phase 3: LoRA Fine-Tuning")
    print("=" * 60)

    # System info
    print(f"\n[SYSTEM] PyTorch: {torch.__version__}")
    print(f"[SYSTEM] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[SYSTEM] GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    # Load processor
    print(f"\n[SETUP] Loading processor: {config['model_name']}")
    processor = WhisperProcessor.from_pretrained(config["model_name"])

    # Set forced decoder IDs for English transcription
    processor.tokenizer.set_prefix_tokens(
        language=config["language"], task=config["task"]
    )

    # Load preprocessed dataset
    print(f"[DATA] Loading preprocessed dataset from: {config['processed_data_dir']}")
    dataset_dict = load_from_disk(config["processed_data_dir"])
    print(f"[DATA] Train: {len(dataset_dict['train'])}, Test: {len(dataset_dict['test'])}")

    # Prepare training features
    dataset_dict = prepare_dataset_for_training(dataset_dict, processor, config)

    # Setup model with quantization + LoRA
    model = setup_quantized_lora_model(config)

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        fp16=config["fp16"],
        gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=config["dataloader_num_workers"],
        remove_unused_columns=config["remove_unused_columns"],
        label_names=config["label_names"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        report_to=config["report_to"],
        predict_with_generate=True,
        generation_max_length=444,
        eval_strategy="steps",
        save_strategy="steps",
        optim="adamw_bnb_8bit",
    )

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        data_collator=data_collator,
        processing_class=processor,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )

    # Train
    print("\n[TRAIN] Starting LoRA fine-tuning...")
    print(f"  Effective batch size: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  Epochs: {config['num_train_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  LoRA rank: {config['lora_r']}, alpha: {config['lora_alpha']}")
    print(f"  Target modules: {config['lora_target_modules']}")

    train_result = trainer.train()

    # Save final LoRA adapters
    final_adapter_dir = os.path.join(config["output_dir"], "final_adapter")
    model.save_pretrained(final_adapter_dir)
    processor.save_pretrained(final_adapter_dir)
    print(f"\n[SAVE] LoRA adapters saved to: {final_adapter_dir}")

    # Save training metrics
    metrics = train_result.metrics
    if torch.cuda.is_available():
        metrics["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    metrics_path = os.path.join(config["output_dir"], "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Training metrics saved to: {metrics_path}")

    # Cleanup
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[DONE] Phase 3 complete. LoRA adapters trained and saved.")
    print(f"  Adapters: {final_adapter_dir}")
    print("\nNext step: Run 3_app_eval.py to evaluate and deploy.")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    train(CONFIG)
