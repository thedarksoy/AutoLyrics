"""
AUTOLYRICS - Script 3: Fine-Tuned Evaluation & Gradio Deployment
=================================================================
Loads base whisper-small + trained LoRA adapters, evaluates on the test split
to measure WER/CER improvement vs. baseline, and launches a Gradio interface
for interactive singing transcription.

Hardware target: RTX 4060 (8GB VRAM)
"""

import os
import re
import json
import time
import gc

import torch
import numpy as np
import torchaudio
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from jiwer import wer, cer
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Model
    "model_name": "openai/whisper-small",
    "adapter_dir": "outputs/whisper-small-lora/final_adapter",
    "language": "en",
    "task": "transcribe",

    # Data
    "processed_data_dir": "data/processed",
    "audio_column": "audio",
    "text_column": "normalized_text",
    "target_sample_rate": 16000,

    # Results
    "baseline_results_file": "outputs/baseline_results.json",
    "finetuned_results_file": "outputs/finetuned_results.json",
    "comparison_file": "outputs/comparison_report.json",

    # Gradio
    "gradio_share": False,
    "gradio_server_port": 7860,
}


# ============================================================================
# TEXT NORMALIZATION (same as Script 1 for consistency)
# ============================================================================
def normalize_text(text: str) -> str:
    """Normalize text for fair WER/CER comparison."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_finetuned_model(config: dict):
    """
    Load base whisper-small and merge LoRA adapters.
    Uses float16 for inference to save VRAM.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[MODEL] Loading base model: {config['model_name']}")
    processor = WhisperProcessor.from_pretrained(config["model_name"])
    model = WhisperForConditionalGeneration.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # Load LoRA adapters
    adapter_dir = config["adapter_dir"]
    if os.path.exists(adapter_dir):
        print(f"[MODEL] Loading LoRA adapters from: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)
        # Merge LoRA weights into the base model for faster inference
        print("[MODEL] Merging LoRA weights for inference...")
        model = model.merge_and_unload()
    else:
        print(f"[WARNING] No adapter found at {adapter_dir}. Using base model only.")

    model.eval()

    if device == "cuda":
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"[MODEL] VRAM after loading: {vram:.2f} GB")

    return model, processor, device


# ============================================================================
# FINE-TUNED EVALUATION
# ============================================================================
def evaluate_finetuned(config: dict) -> dict:
    """
    Run the fine-tuned model on the test split and compute WER/CER.
    Returns metrics for comparison against baseline.
    """
    print("\n" + "=" * 60)
    print("FINE-TUNED MODEL EVALUATION")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, processor, device = load_finetuned_model(config)

    # Use language/task directly in generate() (modern API)

    # Load test data
    print(f"[DATA] Loading test data from: {config['processed_data_dir']}")
    dataset_dict = load_from_disk(config["processed_data_dir"])
    test_ds = dataset_dict["test"]
    audio_col = config["audio_column"]
    text_col = config["text_column"]

    all_references = []
    all_predictions = []
    total_inference_time = 0.0
    total_audio_seconds = 0.0

    print(f"[EVAL] Running inference on {len(test_ds)} test examples...")

    for i in tqdm(range(len(test_ds)), desc="Fine-Tuned Inference"):
        example = test_ds[i]
        audio = example[audio_col]
        audio_array = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        reference = example[text_col]

        if not reference.strip():
            continue

        audio_duration = len(audio_array) / sr
        total_audio_seconds += audio_duration

        # Prepare input
        input_features = processor(
            audio_array, sampling_rate=sr, return_tensors="pt"
        ).input_features

        if device == "cuda":
            input_features = input_features.to(device).half()

        # Inference
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

        # Decode
        prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        prediction = normalize_text(prediction)

        all_references.append(reference)
        all_predictions.append(prediction)

        if i < 3:
            print(f"\n  Example {i}:")
            print(f"    REF: {reference[:100]}")
            print(f"    HYP: {prediction[:100]}")

    # Metrics
    if not all_references:
        print("[EVAL] ERROR: No valid examples found!")
        return {}

    ft_wer = wer(all_references, all_predictions)
    ft_cer = cer(all_references, all_predictions)

    peak_vram = 0.0
    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / 1e9

    avg_latency = total_inference_time / len(all_references)
    rtf = total_inference_time / total_audio_seconds if total_audio_seconds > 0 else 0

    results = {
        "model": config["model_name"],
        "approach": "LoRA fine-tuned (decoder q_proj, v_proj)",
        "adapter_dir": config["adapter_dir"],
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
        "num_test_examples": len(all_references),
        "wer": round(ft_wer, 4),
        "cer": round(ft_cer, 4),
        "avg_latency_seconds": round(avg_latency, 3),
        "real_time_factor": round(rtf, 3),
        "peak_vram_gb": round(peak_vram, 2),
        "total_audio_seconds": round(total_audio_seconds, 1),
        "total_inference_seconds": round(total_inference_time, 1),
    }

    print("\n" + "-" * 40)
    print("FINE-TUNED RESULTS")
    print("-" * 40)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("-" * 40)

    # Save
    os.makedirs(os.path.dirname(config["finetuned_results_file"]), exist_ok=True)
    with open(config["finetuned_results_file"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE] Results saved to: {config['finetuned_results_file']}")

    # Cleanup
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


# ============================================================================
# COMPARISON REPORT
# ============================================================================
def generate_comparison(config: dict):
    """Load baseline and fine-tuned results, print comparison."""
    print("\n" + "=" * 60)
    print("COMPARATIVE ANALYSIS: Baseline vs. Fine-Tuned")
    print("=" * 60)

    baseline_path = config["baseline_results_file"]
    finetuned_path = config["finetuned_results_file"]

    if not os.path.exists(baseline_path):
        print(f"[WARNING] Baseline results not found at {baseline_path}. Run 1_baseline_data.py first.")
        return
    if not os.path.exists(finetuned_path):
        print(f"[WARNING] Fine-tuned results not found at {finetuned_path}.")
        return

    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(finetuned_path) as f:
        finetuned = json.load(f)

    # Calculate improvements
    wer_reduction_abs = baseline["wer"] - finetuned["wer"]
    wer_reduction_rel = (wer_reduction_abs / baseline["wer"]) * 100 if baseline["wer"] > 0 else 0
    cer_reduction_abs = baseline["cer"] - finetuned["cer"]
    cer_reduction_rel = (cer_reduction_abs / baseline["cer"]) * 100 if baseline["cer"] > 0 else 0

    comparison = {
        "baseline_wer": baseline["wer"],
        "finetuned_wer": finetuned["wer"],
        "wer_absolute_reduction": round(wer_reduction_abs, 4),
        "wer_relative_reduction_pct": round(wer_reduction_rel, 2),
        "baseline_cer": baseline["cer"],
        "finetuned_cer": finetuned["cer"],
        "cer_absolute_reduction": round(cer_reduction_abs, 4),
        "cer_relative_reduction_pct": round(cer_reduction_rel, 2),
        "baseline_vram_gb": baseline.get("peak_vram_gb", "N/A"),
        "finetuned_vram_gb": finetuned.get("peak_vram_gb", "N/A"),
        "baseline_latency": baseline.get("avg_latency_seconds", "N/A"),
        "finetuned_latency": finetuned.get("avg_latency_seconds", "N/A"),
        "target_met": wer_reduction_rel >= 15.0,
    }

    print(f"\n  {'Metric':<30} {'Baseline':<15} {'Fine-Tuned':<15} {'Change':<15}")
    print(f"  {'-'*75}")
    print(f"  {'WER':<30} {baseline['wer']:<15.4f} {finetuned['wer']:<15.4f} {wer_reduction_rel:+.2f}%")
    print(f"  {'CER':<30} {baseline['cer']:<15.4f} {finetuned['cer']:<15.4f} {cer_reduction_rel:+.2f}%")
    print(f"  {'Peak VRAM (GB)':<30} {baseline.get('peak_vram_gb', 'N/A'):<15} {finetuned.get('peak_vram_gb', 'N/A'):<15}")
    print(f"  {'Avg Latency (s)':<30} {baseline.get('avg_latency_seconds', 'N/A'):<15} {finetuned.get('avg_latency_seconds', 'N/A'):<15}")
    print(f"\n  TARGET (>15% relative WER reduction): {'YES - MET' if comparison['target_met'] else 'NO - NOT MET'}")

    with open(config["comparison_file"], "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[SAVE] Comparison saved to: {config['comparison_file']}")

    return comparison


# ============================================================================
# GRADIO INTERFACE
# ============================================================================
def launch_gradio(config: dict):
    """Launch an interactive Gradio UI for singing transcription."""
    import gradio as gr

    print("\n" + "=" * 60)
    print("AUTOLYRICS - Gradio Interactive Demo")
    print("=" * 60)

    # Load model once for the Gradio app
    model, processor, device = load_finetuned_model(config)

    def transcribe(audio_path):
        """Transcribe audio input (file upload or microphone recording)."""
        if not audio_path:
            return ""

        try:
            # We strictly use type='filepath' so audio_path is ALWAYS a string path
            # This fixes the numpy int16 normalization bug and ensures uniform processing
            waveform, sr = torchaudio.load(audio_path)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to 16kHz
            if sr != config["target_sample_rate"]:
                resampler = torchaudio.transforms.Resample(sr, config["target_sample_rate"])
                waveform = resampler(waveform)

            audio_array = waveform.squeeze().numpy()

            # Truncate to 30s max
            max_samples = config["target_sample_rate"] * 30
            if len(audio_array) > max_samples:
                audio_array = audio_array[:max_samples]

            # Prepare features
            input_features = processor(
                audio_array,
                sampling_rate=config["target_sample_rate"],
                return_tensors="pt",
            ).input_features

            if device == "cuda":
                input_features = input_features.to(device).half()

            # Generate
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    language=config["language"],
                    task=config["task"],
                    max_new_tokens=444,
                )

            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription.strip()

        except Exception as e:
            return f"Error during transcription: {str(e)}"

    # Polished Custom CSS for a beautiful, premium glassmorphism dark-green theme
    custom_css = """
    body, .gradio-container, .gradio-container-4-20-0 { 
        background: linear-gradient(135deg, #022c22, #064e3b, #0f172a) !important; 
        background-color: #022c22 !important;
        background-attachment: fixed !important;
    }
    .gradio-container { border-radius: 20px; font-family: 'Inter', sans-serif; border: none !important; }
    .glass-panel {
        background: rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(52, 211, 153, 0.2) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
    }
    .main-title {
        background: linear-gradient(to right, #10b981, #34d399, #6ee7b7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        text-align: center;
        font-size: 4rem;
        margin-bottom: 1.5rem;
        padding-top: 1.5rem;
        letter-spacing: 2px;
    }
    .lyrics-output textarea, .lyrics-output textarea::placeholder {
        font-size: 1.6rem !important;
        line-height: 1.8 !important;
        color: #4ade80 !important;
        text-align: center !important;
        font-weight: 600 !important;
        background: transparent !important;
        border: none !important;
        text-shadow: 0 0 10px rgba(74, 222, 128, 0.3);
    }
    .lyrics-output {
        background: rgba(0, 0, 0, 0.6) !important;
    }
    .lyrics-output * {
        background-color: transparent !important;
    }
    """

    with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as demo:
        gr.HTML("<h1 class='main-title'>🎤 AUTOLYRICS</h1>")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Drop a song or record your voice",
                    type="filepath",
                    sources=["upload", "microphone"],
                    elem_classes=["glass-panel"]
                )
            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Transcribed Lyrics",
                    lines=8,
                    placeholder="Singing transcription will appear here magically...",
                    elem_classes=["glass-panel", "lyrics-output"]
                )

        # Trigger auto-transcribe seamlessly when recording stops or file uploads
        audio_input.change(fn=transcribe, inputs=audio_input, outputs=output_text)

    print(f"[GRADIO] Launching on port {config['gradio_server_port']}...")
    demo.launch(
        share=config["gradio_share"],
        server_port=config["gradio_server_port"],
    )


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("AUTOLYRICS - Phase 4 & 5: Evaluation + Deployment")
    print("=" * 60)

    # Phase 4: Evaluate fine-tuned model
    print("\n--- PHASE 4: Fine-Tuned Evaluation ---")
    ft_results = evaluate_finetuned(CONFIG)

    if ft_results:
        generate_comparison(CONFIG)

    # Phase 5: Launch Gradio
    print("\n--- PHASE 5: Interactive Deployment ---")
    launch_gradio(CONFIG)


if __name__ == "__main__":
    main()
