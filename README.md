Emotion Detection in Text Using Logistic Regression and DistilBERT

Overview

This repository contains two implementations for emotion detection in text:

Logistic Regression Model: A traditional machine learning approach using TF-IDF features.

DistilBERT Model: A transformer-based deep learning model fine-tuned for emotion classification.

The goal of this project is to compare the performance of these models and showcase their strengths in detecting emotions such as joy, sadness, anger, and more from textual data.

Table of Contents
Project Features

Dataset

Models

Logistic Regression

DistilBERT

Results

Installation

Usage

Repository Structure

Acknowledgments

Project Features
Preprocessed text data for emotion detection.

Implementation of Logistic Regression with TF-IDF feature extraction.

Fine-tuning of the DistilBERT model on an emotion dataset.

Evaluation metrics such as accuracy, F1-score, and confusion matrix.

A live demo showcasing real-time emotion predictions.

Dataset
The project uses the Emotion Dataset from Hugging Face's datasets library:

Contains labeled text data with six emotions: joy, sadness, anger, fear, love, surprise.

Dataset splits:

Training: 16,000 samples

Validation: 2,000 samples

Test: 2,000 samples

For more details about the dataset, visit the Hugging Face dataset repository.

Models
1. Logistic Regression
Overview: A simple yet effective machine learning model trained on TF-IDF features.

Key Steps:

Text preprocessing (removal of stop words, punctuation).

TF-IDF vectorization to extract numerical features from text.

Training a logistic regression classifier to predict emotions.

Performance:

Accuracy: ~85%

Weighted F1-score: ~84%

2. DistilBERT
Overview: A transformer-based deep learning model fine-tuned for emotion detection.

Key Features:

Tokenization using WordPiece tokenizer.

Attention mechanisms to capture contextual relationships in text.

Fine-tuned on the emotion dataset for two epochs using the AdamW optimizer.

Performance:

Accuracy: ~93.8%

Weighted F1-score: ~92%

Results
Model	Accuracy	F1 Score
Logistic Regression	~85%	~84%
DistilBERT	~93.8%	~92%
Confusion Matrix (DistilBERT):
(Insert a color-coded confusion matrix image here)

Key Insight:
Fine-tuning DistilBERT significantly improved performance compared to a pre-trained model without task-specific training.

Installation
Prerequisites
Ensure you have Python 3.7+ installed along with the following libraries:

transformers

datasets

scikit-learn

numpy

torch

Steps
Clone the repository:

bash
git clone https://github.com/your_username/emotion-detection.git
cd emotion-detection
Install required dependencies:

bash
pip install -r requirements.txt
Usage
Running Logistic Regression Model
Open the lr.ipynb notebook in Jupyter or Google Colab.

Execute the cells to preprocess data, train the logistic regression model, and evaluate its performance.

Running DistilBERT Model
Open the distilbert.ipynb notebook in Jupyter or Google Colab.

Execute the cells to fine-tune DistilBERT on the emotion dataset and evaluate its performance.

Real-Time Prediction Example
To test a custom sentence using DistilBERT:

python
text = "I am feeling very happy today!"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
print(f"Predicted Emotion: {classes[predicted_class_id]}")
Repository Structure
text
emotion-detection/
├── lr.ipynb                # Notebook for Logistic Regression implementation
├── distilbert.ipynb        # Notebook for DistilBERT implementation
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation (this file)
Acknowledgments
Special thanks to:

Hugging Face for providing pre-trained models and datasets.

Scikit-learn for its robust machine learning tools.

The open-source community for their invaluable resources and contributions.

Feel free to explore the notebooks and provide feedback or suggestions!
