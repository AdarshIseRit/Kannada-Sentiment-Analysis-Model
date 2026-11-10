# KannadaSentiment – Hybrid MuRIL + TF-IDF Kannada Emotion Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)

> 🚀 A hybrid MuRIL + TF-IDF based emotion detection system for Kannada text,
> capable of classifying six emotions (happy, sad, anger, fear, disgust, neutral)
> through an interactive Streamlit web interface.

KannadaSentiment is an end-to-end emotion detection system for Kannada text.
It combines modern transformer-based language understanding with classical
machine learning to accurately classify six human emotions from natural language input.

## Planned Architecture
- **Data Agent**: Preprocess Kannada text and handle imbalanced distributions.
- **Feature Agent**: Extract TF-IDF n-grams and MuRIL transformer embeddings.
- **Training Agent**: Train and optimize hybrid models with Logistic Regression & MuRIL.
- **Evaluation Agent**: Benchmark validation metrics and generate confusion matrix.
- **Prediction Agent**: Real-time inference engine for Kannada phrases.
- **Streamlit Dashboard**: Interactive web app with real-time analysis and batch spreadsheet inference.
