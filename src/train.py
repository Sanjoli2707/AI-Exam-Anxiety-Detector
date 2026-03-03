# src/train.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Dummy dataset example
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Initialize
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Dummy data
texts = ["I am anxious", "Feeling calm"]
labels = [1, 0]  # 1 = anxious, 0 = calm
dataset = TextDataset(texts, labels, tokenizer)
loader = DataLoader(dataset, batch_size=2)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(2):  # small number for testing
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'])
        loss = outputs.loss
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

# Save model
torch.save(model.state_dict(), '../model/bert_anxiety_model.pt')
print("Model saved!")