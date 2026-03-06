from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertForSequenceClassification, BertTokenizer

app = FastAPI()

model_path = "../model/bert_anxiety_model"

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

model.eval()

labels = ['Anxiety','Depression','Normal','Suicidal']


class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "AI Exam Anxiety Detector API is running"}


@app.post("/predict")
def predict(request: TextRequest):

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits).item()

    return {"prediction": labels[pred]}