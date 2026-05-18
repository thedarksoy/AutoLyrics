import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

def export_model():
    print("============================================================")
    print("AUTOLYRICS - Exporting Full Model")
    print("============================================================")
    
    base_model_id = "openai/whisper-small"
    adapter_path = "outputs/whisper-small-lora/final_adapter"
    export_path = "outputs/autolyrics-model-full"

    print(f"[SETUP] Loading processor: {base_model_id}")
    processor = WhisperProcessor.from_pretrained(base_model_id)
    
    print(f"[MODEL] Loading base model on CPU: {base_model_id}")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map="cpu"
    )
    
    print(f"[LORA] Loading PEFT adapters from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("[MERGE] Merging adapters into base weights...")
    merged_model = peft_model.merge_and_unload()
    
    print(f"[SAVE] Saving fully merged model to: {export_path}")
    os.makedirs(export_path, exist_ok=True)
    merged_model.save_pretrained(export_path)
    processor.save_pretrained(export_path)
    
    print("[DONE] Export complete! The standalone model is ready for deployment.")

if __name__ == "__main__":
    export_model()
