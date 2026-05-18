# AUTOLYRICS

**Studio-Grade Singing Voice Transcription powered by LoRA Fine-Tuned Whisper**

AUTOLYRICS is a machine learning project dedicated to accurately transcribing singing voices into text (lyrics). By fine-tuning OpenAI's Whisper model on singing datasets using Low-Rank Adaptation (LoRA), this project substantially reduces the Word Error Rate (WER) compared to standard zero-shot speech-to-text models.

## Features
- **Highly Accurate Singing Transcription**: Achieves a massive **59.7% relative improvement** in Word Error Rate over the zero-shot Whisper baseline.
- **Efficient Fine-Tuning**: Uses 8-bit quantization and LoRA targeting attention matrices to train efficiently on consumer hardware (e.g., RTX 4060 8GB).
- **Interactive UI**: dark-themed Gradio web interface for seamless microphone recording or file uploads.

---

##  Performance Report
In our fine-tuning process, we aimed to exceed a target of a 15% relative WER improvement.

| Metric | Zero-Shot Baseline | LoRA Fine-Tuned | Relative Improvement |
| :--- | :--- | :--- | :--- |
| **Word Error Rate (WER)** | 0.8024 | **0.3234** | **59.70%** 📉 |
| **Character Error Rate (CER)** | 0.5475 | **0.1078** | **80.31%** 📉 |
| **Peak VRAM (GB)** | 1.10 | 0.56 | 49.09% 📉 |

*For complete details on data preprocessing, text normalization, and hyperparameter configuration, please see the [REPORT.md](REPORT.md) file.*

---

##  Setup & Installation

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support (for GPU acceleration)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/autolyrics.git
   cd autolyrics
   ```

2. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Preprocess Baseline Data
Downloads the singing dataset, processes audio and text, and evaluates the baseline zero-shot model.
```bash
python 1_baseline_data.py
```

### 2. Train LoRA Adapters
Fine-tunes the base Whisper model using 8-bit quantization and LoRA.
```bash
python 2_train_lora.py
```

### 3. Evaluate & Deploy
Evaluates the fine-tuned LoRA adapters and launches the interactive Gradio demo.
```bash
python 3_app_eval.py
```

---

##  Demo Video
A demonstration of the AUTOLYRICS interface transcribing a small audio clip is included in the release artifacts.

##  License
This project is licensed under the MIT License.
