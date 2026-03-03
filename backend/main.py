# main.py / inference.py
import os
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from fastapi import FastAPI
from pydantic import BaseModel

# =========================
# Device setup
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =========================
# Load tokenizer
# =========================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# =========================
# Define model architecture
# =========================
class BertAnxietyClassifier(nn.Module):
    def __init__(self, num_classes=4):  # changed to 4 classes
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.classifier(x)

# =========================
# Load trained model
# =========================
num_classes = 4  # match the checkpoint
model = BertAnxietyClassifier(num_classes=num_classes)
# Safe path handling
DRIVE_PATH = r"I:\My Drive\AI-Anxiety-Data\bert_anxiety_model.pt"
RELATIVE_PATH = "model/bert_anxiety_model.pt"

if os.path.exists(DRIVE_PATH):
    MODEL_PATH = DRIVE_PATH
elif os.path.exists(RELATIVE_PATH):
    MODEL_PATH = RELATIVE_PATH
else:
    raise FileNotFoundError(
        f"Model not found. Checked:\n1. {DRIVE_PATH}\n2. {RELATIVE_PATH}"
    )

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================
# FastAPI setup
# =========================
app = FastAPI(title="AI-Based Exam Anxiety Detector")

class TextRequest(BaseModel):
    text: str

# =========================
# Prediction endpoint
# =========================
@app.post("/predict")
def predict_anxiety(request: TextRequest):
    # Tokenize input
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Disable gradients for inference
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        pred = torch.argmax(logits, dim=1).item()

    # Map integer prediction to label
    label_map = {
        0: "Low Anxiety",
        1: "Moderate Anxiety",
        2: "High Anxiety",
        3: "Extra Class"  # replace with  actual 4th class name
    }
    return {"anxiety_level": label_map[pred]}

# =========================
# Test with uvicorn:
# python -m uvicorn backend.main:app --reload --port 8501
# =========================