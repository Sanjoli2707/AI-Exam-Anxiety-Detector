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

# GPU and Training Environment

Model training and experimentation are performed using Google Colab to leverage GPU acceleration.
The environment is verified for CUDA availability to support transformer-based model training.

## Version Control Setup

Git and GitHub are configured for version control and collaborative development.
Large files such as datasets and trained model weights are excluded using .gitignore to maintain repository hygiene.

## Dataset Selection and Organization
1) Dataset Collection

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

2) DataSet Loading
To inspect the dataset, a Jupyter Notebook named

01_dataset_inspection.ipynb is created inside the notebooks/ folder.

The dataset is loaded using the Pandas library:

import pandas as pd

df = pd.read_csv("../data/final_anxiety_dataset.csv")

This allows structured analysis and easy visualization of dataset properties.

3) Understanding DataSet Structure
The dataset mainly consists of two important columns:

statement → Text input written by users

status → Mental health label associated with the text

To understand the dataset size and structure, the following commands are used:

df.head()

df.shape

df.columns

This step confirms:

Number of records

Number of columns

Type of data present.

4) Finding Missing Vaues
Before training any model, missing values must be identified.

df.isnull().sum()

This step reveals that some text entries may be missing and need to be handled during preprocessing.

5) Class Distribution Analysis
   The dataset contains multiple mental-health categories, which may not be evenly distributed.

df['status'].value_counts()

This analysis helps identify:

Class imbalance

Dominant mental-health categories

Need for label mapping in later stages

Understanding class distribution is essential to prevent model bias.

# Data Processing & Label Mapping

# 3.2 Original Dataset Labels

The original dataset consists of multiple mental-health categories representing different emotional and psychological states, including:

Normal

Anxiety

Stress

Depression

Suicidal

Bipolar

Personality Disorder

These labels reflect clinical or psychological conditions, whereas the goal of this project is to analyze exam-related anxiety levels, not to perform medical diagnosis.

# 3.3 Custom Label Mapping Strategy

A custom label-mapping strategy is designed based on emotional intensity and severity.

This mapping converts complex mental-health categories into a simpler, practical anxiety scale consisting of three levels:

Low Anxiety

Moderate Anxiety

High Anxiety

The mapping ensures that the classification output aligns with real-world academic stress scenarios while remaining ethically responsible and interpretable.

Label Mapping Table
Original Mental-Health Label	Anxiety Level
Normal	Low Anxiety
Stress	Moderate Anxiety
Anxiety	Moderate Anxiety
Depression	High Anxiety
Suicidal	High Anxiety
Bipolar	High Anxiety
Personality Disorder	High Anxiety
Rationale Behind the Mapping

Low Anxiety indicates emotionally stable or neutral states.

Moderate Anxiety captures common exam-related stress and nervousness.

High Anxiety represents severe emotional distress that can significantly affect academic performance.

This strategy reduces classification complexity while preserving emotional severity and relevance to exam anxiety detection.

# BERT Model Selection & Training Colab(Google Colab)
The BERT (Bidirectional Encoder Representations from Transformers) model is selected for this project due to its strong performance in contextual text understanding.

Reasons for choosing BERT-Base (bert-base-uncased):

Bidirectional understanding of text

Pretrained on large-scale English corpora

Suitable for text classification tasks

Computationally efficient compared to larger variants

BERT enables the system to understand context, emotional cues, and sentence meaning, which is critical for detecting exam anxiety.

Training BERT models requires GPU acceleration, which may not be available or efficient on local systems. Therefore, Google Colab is used for training because:

Provides free GPU support

Reduces training time significantly

Eliminates local hardware limitations

Ensures reproducibility of training

# 4.4 — Text Preprocessing with BERT Tokenizer

Before the model can understand text, raw strings need to be converted into numbers that BERT can work with. Here’s what happens:

Token IDs → Every word/subword is converted into a number corresponding to BERT’s vocabulary.

Attention Masks → Helps the model know which tokens are real and which are just padding.

Truncation/Padding → Makes all sequences the same length (so batches can be processed efficiently).

 Why this is important: BERT can’t work with plain strings—it expects fixed-size numerical inputs. This step ensures the data is compatible with the model.

# 4.5 — Training Setup (train.py)

Once tokenized, the data enters the training loop implemented in PyTorch. Key points:

Component	Details
Loss Function	Cross-Entropy Loss → standard for classification tasks
Optimizer	AdamW → popular for transformer models, includes weight decay for better generalization
Batch Size	Depends on GPU memory (T4 GPU is good for medium batch sizes)
Epochs	Multiple passes over the dataset → helps model learn better
Goal	The model learns to map linguistic patterns to anxiety levels

 Tip: Always monitor loss & accuracy per epoch so you know if the model is actually learning. If loss plateaus or spikes, tweak learning rate or batch size.

 #  Inference & Real-Time Prediction

# Activity 5.1: Switching Model to Inference Mode

Goal: After training, we only need the model for predictions, not for learning.

Key Steps:

Load the saved model (bert_anxiety_model.pt) and tokenizer.

Switch the model to evaluation mode using model.eval().

Disable gradient calculations with torch.no_grad() to save memory and speed up inference.

Outcome:

Model no longer updates weights.

Predictions are stable and efficient.

Interview explanation: “We freeze the model and turn off gradients to ensure fast, memory-efficient predictions during inference.”

# Activity 5.2: Real-Time Prediction Testing

Goal: Test how the model performs on real, manually written exam-related sentences.

Key Steps:

Write example inputs: calm/confident, nervous/stressed, highly anxious/panic-driven.

Tokenize and encode the sentences.

Pass them through the model to get predicted anxiety levels.

Map predicted indices back to labels (Low, Moderate, High).

Outcome:

Observed that predictions match human intuition.

Validated the model’s ability to differentiate between different anxiety levels.

Interview explanation: “We verified the model on real-world exam-related text to ensure it aligns with human judgment.”

# Activity 5.3: Backend API Validation

Goal: Integrate the trained model with the backend and make it accessible to a frontend.

Key Steps:

Load the model in FastAPI backend.

Expose an endpoint (e.g., /predict) that accepts text input.

Test via Swagger UI and Streamlit frontend.

Outcome:

Backend loads the model correctly.

API returns valid predictions.

Frontend receives and displays predictions properly.

Interview explanation: “We ensured that the ML model works in real-time when called from an API and integrates smoothly with the frontend.”

# Activity 5.4: Output Consistency Analysis

Goal: Confirm the model’s predictions are consistent and logical.

Key Steps:

Use multiple test sentences to cover different anxiety patterns.

Check that similar inputs give similar predictions.

Ensure predictions progress logically from Low → Moderate → High anxiety.

Outcome:

Predictions are mostly stable.

Minor variations are expected due to overlapping emotional language.

Model can be considered reliable for deployment.


# Backend Development

The objective of this milestone is to develop a robust backend system for the AI-Based Exam Anxiety Detector. The backend is responsible for loading the trained BERT model, processing user input text, and returning the predicted anxiety level through an API. FastAPI was selected for backend development due to its high performance, simplicity, and built-in API documentation features.

# 6.1: Selecting FastAPI and Folder Structure

FastAPI was chosen as the backend framework because of its speed, scalability, and compatibility with machine learning models. A structured project layout was created to clearly separate backend logic, model files, datasets, and frontend components, improving maintainability and readability.

# 6.2: Loading the Trained BERT Model

The trained BERT-based anxiety classification model is loaded when the server starts. The tokenizer and model weights are initialized, moved to the appropriate device (CPU/GPU), and set to evaluation mode to ensure efficient and stable inference.

# 6.3: Defining Request and Response Schema

Using Pydantic, request and response schemas were defined to ensure structured communication between the frontend and backend. The API accepts text input from users and returns the predicted anxiety category in JSON format.

# 6.4: Implementing the Prediction Endpoint

A /predict API endpoint was implemented to process user input. The endpoint tokenizes the input text using the BERT tokenizer, passes it through the trained model, and returns the predicted anxiety level such as Low, Moderate, or High Anxiety.

# 6.5: Running the Backend Server

The backend server is executed using Uvicorn, which runs the FastAPI application locally and enables real-time communication with the frontend interface.

# 6.6: Testing the Backend Using Swagger UI

FastAPI automatically generates interactive API documentation through Swagger UI, allowing developers to test the /predict endpoint directly in the browser and verify the model’s predictions.

The objective of this milestone is to build a simple and user-friendly interface that allows users to interact with the AI-Based Exam Anxiety Detector. The frontend enables users to enter exam-related thoughts, submit them for analysis, and instantly view the predicted anxiety level along with supportive feedback.

# Connecting Frontend to Backend API
The Streamlit frontend sends a POST request to the FastAPI /predict endpoint when the user clicks the Predict button. The backend processes the text using the trained BERT model and returns the prediction in JSON format, enabling real-time communication between the frontend and backend.

Displaying Prediction Results
The predicted anxiety level (Low, Moderate, or High) is displayed clearly on the interface with appropriate emojis to improve user engagement and visual understanding.

Anxiety Management Tips
Based on the prediction, the interface provides simple tips such as breathing exercises, time management suggestions, and reassurance messages to help users manage exam stress.

Running the Frontend
The Streamlit interface can be launched using:

streamlit run frontend/app.py

Once started, the application opens in the browser and allows users to test the system in real time.