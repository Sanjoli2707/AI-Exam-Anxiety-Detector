## AI Exam Anxiety Detector

An end-to-end Natural Language Processing (NLP) system designed to detect exam-related anxiety from textual inputs using transformer-based deep learning models. The project focuses on identifying anxiety patterns from mental-health text data and provides a deployable demo interface for real-time predictions.

## Project Objectives

Analyze text-based mental-health data to identify anxiety-related patterns

Leverage transformer-based models (BERT) for accurate text classification

Build a clean and modular ML pipeline from data preprocessing to deployment

Provide a user-friendly frontend for real-time anxiety detection

## Environment Setup and Configuration
1) Python Installation and Virtual Environment

Python version 3.9 or above is used for this project.
A dedicated virtual environment is created to ensure isolated and clean dependency management.

python -m venv venv
venv\Scripts\activate
2) Library Installation

All required libraries for NLP processing, BERT model integration, backend services, and frontend UI are installed using a centralized requirements.txt file.

pip install -r requirements.txt
3) Environment Validation

The virtual environment is validated to ensure successful installation of key libraries including:

PyTorch

Hugging Face Transformers

FastAPI

Streamlit

This confirms readiness for model training, API development, and UI rendering.

## Project Directory Structure

A modular and scalable project structure is maintained to separate concerns clearly.

AI-EXAM-ANXIETY-DETECTOR/
│
├── backend/        # Backend API services
├── frontend/       # Streamlit-based user interface
├── src/            # Core ML logic (preprocessing, training, modeling)
├── data/           # Dataset directory (ignored in GitHub)
├── model/          # Trained model storage (ignored in GitHub)
│
├── requirements.txt
├── .gitignore
└── README.md
 GPU and Training Environment

Model training and experimentation are performed using Google Colab to leverage GPU acceleration.
The environment is verified for CUDA availability to support transformer-based model training.

## Version Control Setup

Git and GitHub are configured for version control and collaborative development.
Large files such as datasets and trained model weights are excluded using .gitignore to maintain repository hygiene.

## Dataset Selection and Organization
Dataset Collection

A publicly available mental-health text dataset was obtained from Kaggle. The dataset contains thousands of text statements annotated with emotional and psychological categories including:

Normal

Anxiety

Stress

Depression

Suicidal

Among the available dataset files, mental_health_combined.csv was selected as the primary dataset as it consolidates all categories into a single structured file suitable for transformer-based text classification.

Additional dataset files were used only for analytical understanding:

An unbalanced dataset to study class distribution

A feature-engineered dataset not required for BERT-based modeling

Note: Dataset files are excluded from this repository due to licensing and size constraints. Users must download the dataset separately from Kaggle and place the required CSV file inside the data/ directory.
