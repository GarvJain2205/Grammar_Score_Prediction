# Spoken Grammar Scoring Engine 🎙️🤖

## Project Overview

This project builds an **AI-based grammar scoring system for spoken audio**.
The model predicts a **grammar score between 0 and 5** from speech recordings.

Two different pipelines were implemented:

1. **Audio-based pipeline** using Wav2Vec2 embeddings and machine learning models.
2. **Text-based pipeline** using Whisper speech-to-text and a BERT-based grammar scorer.

The goal is to evaluate **spoken language grammar quality automatically using deep learning and NLP techniques**.

---

# Dataset

Dataset used: **SHL Hiring Test Speech Grammar Scoring Dataset**

The dataset contains:

* `.wav` audio recordings of spoken responses
* Metadata CSV with filenames and grammar score labels

Example structure:

```
dataset/
 ├── train.csv
 ├── test.csv
 ├── audios_train/
 └── audios_test/
```

Each audio file is associated with a **grammar score label (0–5)**.

---

# Approach 1: Audio Embedding Pipeline

### Steps

1. Audio files are loaded and resampled to **16 kHz**
2. **Wav2Vec2 pretrained model** extracts speech embeddings
3. Feature vectors are standardized
4. Machine learning regression models predict grammar scores

Models tested:

* Linear Regression
* Ridge Regression
* Random Forest
* Gradient Boosting
* SVR
* XGBoost

Best performance:

**Wav2Vec2 + SVR**

Performance metric:

```
Pearson Correlation: 0.772
```

---

# Approach 2: Speech-to-Text + NLP Pipeline

### Steps

1. Audio files are transcribed using **OpenAI Whisper**
2. Transcriptions are tokenized using **BERT tokenizer**
3. A **BERT-based regression model** predicts grammar scores

Pipeline:

```
Audio → Whisper ASR → Text → BERT → Grammar Score
```

---

# Model Architecture

### Audio Pipeline

```
Audio (.wav)
   ↓
Wav2Vec2 Embeddings
   ↓
Feature Scaling
   ↓
SVR Regression Model
   ↓
Grammar Score Prediction
```

### NLP Pipeline

```
Audio (.wav)
   ↓
Whisper Speech-to-Text
   ↓
BERT Tokenization
   ↓
BERT Regression Model
   ↓
Grammar Score Prediction
```

---

# Technologies Used

* Python
* PyTorch
* HuggingFace Transformers
* Wav2Vec2
* OpenAI Whisper
* BERT
* Scikit-learn
* XGBoost
* Librosa
* Torchaudio

---

# Evaluation Metrics

Models were evaluated using:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* R² Score
* Pearson Correlation

Best result:

```
Pearson Correlation: 0.772
```

---

# Future Improvements

* Fine-tuning Wav2Vec2 on grammar scoring data
* Using larger transformer models
* Combining audio and text features
* Building a real-time speech evaluation system

---

# Author

**Garv Jain**
Computer Science Undergraduate
Focus: AI, NLP, and Speech Processing
I can also help you add **one small section that will make your GitHub project look “research-level” and impress recruiters immediately**.
