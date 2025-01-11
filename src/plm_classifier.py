from typing import List, Dict
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
from transformers import logging

warnings.filterwarnings('ignore', message='Some weights of')
warnings.filterwarnings('ignore', message='Torch was not compiled with flash attention')
logging.set_verbosity_error()

class RestaurantAspectDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[Dict] = None, tokenizer=None, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"Positive": 0, "Négative": 1, "Neutre": 2, "NE": 3}
        self.aspects = ["Prix", "Cuisine", "Service", "Ambiance"]
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        try:
            text = self.texts[idx]
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            item = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }
            
            if self.labels is not None:
                label_tensor = torch.zeros(len(self.aspects), dtype=torch.long)
                for i, aspect in enumerate(self.aspects):
                    try:
                        label_tensor[i] = self.label_map[self.labels[idx][aspect]]
                    except KeyError as e:
                        print(f"Error processing label for {aspect}: {self.labels[idx][aspect]}")
                        raise e
                item['labels'] = label_tensor
                
            return item
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            raise e

class PLMClassifier:
    def __init__(self, device: int):
        self.device = device if device >= 0 else "cpu"
        self.model_name = "almanach/camembert-base"
        self.tokenizer = CamembertTokenizer.from_pretrained(self.model_name)
        self.num_labels = 4  # Positive, Négative, Neutre, NE
        self.aspects = ["Prix", "Cuisine", "Service", "Ambiance"]
        self.label_map_reverse = {0: "Positive", 1: "Négative", 2: "Neutre", 3: "NE"}
        
        # Initialize model for each aspect
        print("Initializing models for each aspect...")
        self.models = {}
        for aspect in self.aspects:
            model = CamembertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )
            self.models[aspect] = model.to(self.device)
            
    def train(self, train_data: List[Dict], val_data: List[Dict], batch_size=32, epochs=5):
        print("\nPreparing datasets...")
        train_start_time = time.time()
        
        train_texts = [item['Avis'] for item in train_data]
        train_labels = [{k:v for k,v in item.items() if k in self.aspects} for item in train_data]
        
        val_texts = [item['Avis'] for item in val_data]
        val_labels = [{k:v for k,v in item.items() if k in self.aspects} for item in val_data]
        
        train_dataset = RestaurantAspectDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = RestaurantAspectDataset(val_texts, val_labels, self.tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        metrics = {'train_time': 0, 'val_accuracies': {}}
        
        # Training loop for each aspect
        for aspect_idx, aspect in enumerate(self.aspects):
            print(f"\nTraining classifier for {aspect}...")
            model = self.models[aspect]
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            best_val_accuracy = 0
            patience = 3
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                total_loss = 0
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
                for batch in progress_bar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'][:, aspect_idx].to(self.device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                # Validation phase
                val_accuracy = self._evaluate_aspect(model, val_loader, aspect_idx)
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Average training loss: {total_loss/len(train_loader):.4f}")
                print(f"Validation accuracy: {val_accuracy:.2f}%")
                
                # Early stopping
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered for {aspect}")
                        break
            
            metrics['val_accuracies'][aspect] = best_val_accuracy
        
        metrics['train_time'] = time.time() - train_start_time
        print("\nTraining completed!")
        print(f"Total training time: {metrics['train_time']:.2f} seconds")
        print("\nValidation accuracies:")
        for aspect, acc in metrics['val_accuracies'].items():
            print(f"{aspect}: {acc:.2f}%")
        print(f"Average accuracy: {np.mean(list(metrics['val_accuracies'].values())):.2f}%")
        
        return metrics
                
    
    def _evaluate_aspect(self, model, data_loader, aspect_idx):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'][:, aspect_idx].cpu().numpy()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                
                all_preds.extend(predictions)
                all_labels.extend(labels)
        
        accuracy = accuracy_score(all_labels, all_preds) * 100
        return accuracy
    
    def predict(self, texts: List[str], batch_size=32) -> List[Dict]:
        print("Running predictions...")
        predict_start_time = time.time()
        
        dataset = RestaurantAspectDataset(texts, labels=None, tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        total_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            batch_predictions = {}
            
            for aspect in self.aspects:
                model = self.models[aspect]
                model.eval()
                
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=1)
                    
                batch_predictions[aspect] = [
                    self.label_map_reverse[pred.item()]
                    for pred in predictions
                ]
            
            for i in range(len(batch_predictions[self.aspects[0]])):
                pred_dict = {
                    aspect: batch_predictions[aspect][i]
                    for aspect in self.aspects
                }
                all_predictions.append(pred_dict)
                
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{total_batches} batches")
        
        predict_time = time.time() - predict_start_time
        print(f"\nPrediction completed in {predict_time:.2f} seconds")
        print(f"Average time per sample: {predict_time/len(texts)*1000:.2f} ms")
        
        return all_predictions