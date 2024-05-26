import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import tkinter as tk
from tkinter import simpledialog
import os
import logging

# Logger ayarı
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(train_file_path, test_file_path):
    if not os.path.isfile(train_file_path):
        raise FileNotFoundError(f"{train_file_path} dosyası bulunamadı.")
    if not os.path.isfile(test_file_path):
        raise FileNotFoundError(f"{test_file_path} dosyası bulunamadı.")
    
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    
    return train_data, test_data

def preprocess_data(train_data, test_data):
    # İlgili sütunları al
    train_texts = train_data.iloc[:, 3]
    train_labels = train_data.iloc[:, 2]
    test_texts = test_data.iloc[:, 3]
    test_labels = test_data.iloc[:, 2]

    # Etiketleri sayısal değerlere dönüştür
    label_mapping = {"positive": 1, "neutral": 0, "negative": -1}
    train_labels = train_labels.map(label_mapping)
    test_labels = test_labels.map(label_mapping)

    # NaN değerlerini temizle
    train_data = train_data.dropna(subset=[train_data.columns[2], train_data.columns[3]])
    test_data = test_data.dropna(subset=[test_data.columns[2], test_data.columns[3]])

    train_texts = train_data.iloc[:, 3]
    train_labels = train_data.iloc[:, 2]
    test_texts = test_data.iloc[:, 3]
    test_labels = test_data.iloc[:, 2]
    
    return train_texts, train_labels, test_texts, test_labels

def train_svm_model(train_texts, train_labels, test_texts, test_labels):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()
    
    svm_model = SVC()
    svm_model.fit(X_train, train_labels)
    svm_predictions = svm_model.predict(X_test)
    
    accuracy = accuracy_score(test_labels, svm_predictions)
    precision = precision_score(test_labels, svm_predictions, average='weighted')
    recall = recall_score(test_labels, svm_predictions, average='weighted')
    f1 = f1_score(test_labels, svm_predictions, average='weighted')
    
    logger.info(f"SVM Model - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    
    return vectorizer, X_train, X_test, svm_model

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_alexnet_model(X_train, train_labels, X_test, test_labels):
    input_size = X_train.shape[1]
    hidden_size = 500
    output_size = 3
    alex_model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(alex_model.parameters(), lr=0.001)

    train_data_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.long)
    test_data_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels.values, dtype=torch.long)

    num_epochs = 10
    batch_size = 64

    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for i, (texts, labels) in enumerate(train_loader):
            outputs = alex_model(texts)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    alex_model.eval()
    with torch.no_grad():
        outputs = alex_model(test_data_tensor)
        _, predicted = torch.max(outputs.data, 1)
        total = test_labels_tensor.size(0)
        correct = (predicted == test_labels_tensor).sum().item()

        accuracy = 100 * correct / total
        logger.info(f'AlexNet Accuracy: {accuracy} %')
    
    return alex_model

def encode_texts(texts, tokenizer, max_length=512):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

def train_bert_model(train_texts, train_labels, test_texts, test_labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    
    train_inputs, train_masks = encode_texts(train_texts, tokenizer)
    test_inputs, test_masks = encode_texts(test_texts, tokenizer)

    train_labels_tensor = torch.tensor(train_labels.values)
    test_labels_tensor = torch.tensor(test_labels.values)

    batch_size = 16

    train_data = TensorDataset(train_inputs, train_masks, train_labels_tensor)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_masks, test_labels_tensor)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    optimizer = AdamW(bert_model.parameters(), lr=2e-5)
    epochs = 4

    for epoch in range(epochs):
        bert_model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_labels = batch
            bert_model.zero_grad()
            outputs = bert_model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_dataloader):.4f}')

    bert_model.eval()
    predictions, true_labels = []

    for batch in test_dataloader:
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = bert_model(b_input_ids, attention_mask=b_input_mask)
        
        logits = outputs.logits
        predictions.append(logits)
        true_labels.append(b_labels)

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = torch.argmax(torch.stack(flat_predictions), dim=1).cpu().numpy()
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    accuracy = accuracy_score(flat_true_labels, flat_predictions)
    precision = precision_score(flat_true_labels, flat_predictions, average='weighted')
    recall = recall_score(flat_true_labels, flat_predictions, average='weighted')
    f1 = f1_score(flat_true_labels, flat_predictions, average='weighted')

    logger.info(f'BERT Model - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

    return tokenizer, bert_model

def predict_sentiment(text, vectorizer, svm_model, alex_model, tokenizer, bert_model):
    svm_input = vectorizer.transform([text]).toarray()
    svm_prediction = svm_model.predict(svm_input)[0]

    alex_input = torch.tensor(svm_input, dtype=torch.float32)
    alex_output = alex_model(alex_input)
    _, alex_prediction = torch.max(alex_output.data, 1)
    alex_prediction = alex_prediction.item()

    bert_input, bert_mask = encode_texts([text], tokenizer)
    with torch.no_grad():
        bert_output = bert_model(bert_input, attention_mask=bert_mask)
    bert_prediction = torch.argmax(bert_output.logits, dim=1).item()

    return svm_prediction, alex_prediction, bert_prediction

def get_sentiment_label(pred):
    return 'positive' if pred == 1 else 'neutral' if pred == 0 else 'negative'

def on_submit(vectorizer, svm_model, alex_model, tokenizer, bert_model, result_text):
    user_input = simpledialog.askstring("Input", "Please enter a sentence:")
    if user_input:
        svm_pred, alex_pred, bert_pred = predict_sentiment(user_input, vectorizer, svm_model, alex_model, tokenizer, bert_model)
        result_text.set(f"SVM: {get_sentiment_label(svm_pred)}\n"
                        f"AlexNet: {get_sentiment_label(alex_pred)}\n"
                        f"BERT: {get_sentiment_label(bert_pred)}")

def main():
    train_file_path = "twitter_training.csv"
    test_file_path = "twitter_validation.csv"

    logger.info("Veri yükleniyor...")
    train_data, test_data = load_data(train_file_path, test_file_path)

    logger.info("Veri önişleniyor...")
    train_texts, train_labels, test_texts, test_labels = preprocess_data(train_data, test_data)

    logger.info("SVM modeli eğitiliyor...")
    vectorizer, X_train, X_test, svm_model = train_svm_model(train_texts, train_labels, test_texts, test_labels)
    
    logger.info("AlexNet modeli eğitiliyor...")
    alex_model = train_alexnet_model(X_train, train_labels, X_test, test_labels)

    logger.info("BERT modeli eğitiliyor...")
    tokenizer, bert_model = train_bert_model(train_texts, train_labels, test_texts, test_labels)

    logger.info("Tkinter arayüzü başlatılıyor...")
    root = tk.Tk()
    root.title("Sentiment Analysis")

    result_text = tk.StringVar()

    label = tk.Label(root, text="Sentiment Analysis Results:")
    label.pack()

    result_label = tk.Label(root, textvariable=result_text)
    result_label.pack()

    button = tk.Button(root, text="Enter a Sentence", command=lambda: on_submit(vectorizer, svm_model, alex_model, tokenizer, bert_model, result_text))
    button.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
