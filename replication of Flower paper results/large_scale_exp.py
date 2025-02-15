import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, AdamW
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and Preprocess Dataset

class AmazonReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load dataset (replace with actual dataset path)
# Assume CSV columns: 'review_text', 'rating'
df = pd.read_csv('Books_rating.csv')
texts = df['review/text'].values
labels = df['review/score'].values - 1  # Convert 1-5 to 0-4 for classification

# Split into train/test (adjust ratios as needed)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Simulate Federated Clients

def create_clients(train_texts, train_labels, num_clients=1000):
    # Split data into num_clients partitions (simulate non-IID)
    client_data = []
    for _ in range(num_clients):
        indices = np.random.choice(len(train_texts), size=1000, replace=True)  # 1k samples/client
        client_texts = train_texts[indices]
        client_labels = train_labels[indices]
        client_dataset = AmazonReviewsDataset(client_texts, client_labels, tokenizer)
        client_data.append(client_dataset)
    return client_data

clients = create_clients(train_texts, train_labels, num_clients=1000000)  # 1M clients

# Define Model & Federated Functions
def initialize_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=5  # 5-class classification (ratings 1-5)
    )
    return model

def train_client(model, dataset, epochs=1, device='cuda'):
    model.train()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    for _ in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    return model.state_dict()

def fed_avg(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = torch.stack([client_model[key] for client_model in client_models], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

# Federated Training Loop
accuracies = []
def federated_learning(num_rounds=10, clients_per_round=100, device='cuda'):
    # Initialize global model
    global_model = initialize_model()
    test_dataset = AmazonReviewsDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)

    for round in range(num_rounds):
        print(f"Round {round + 1}/{num_rounds}")
        # Sample clients
        selected_clients = np.random.choice(clients, size=clients_per_round, replace=False)
        
        # Train clients in parallel (simplified; use multiprocessing in practice)
        client_models = []
        for client_dataset in selected_clients:
            client_model = initialize_model()
            client_model.load_state_dict(global_model.state_dict())
            client_model_state = train_client(client_model, client_dataset, epochs=1, device=device)
            client_models.append(client_model_state)
        
        # Aggregate using FedAvg
        global_model = fed_avg(global_model, client_models)
        
        # Evaluate on test set
        accuracy = evaluate(global_model, test_loader, device)
        print(f"Round {round + 1} Test Accuracy: {accuracy:.2f}%")
        accuracies.append(accuracy)

def evaluate(model, dataloader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return (correct / total) * 100

# Run Experiment

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    federated_learning(num_rounds=60, clients_per_round=100, device=device);

plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.show()
