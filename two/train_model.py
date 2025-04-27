import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

# Define a custom dataset class
class TwitterSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Function to train the model
def train_model(model, train_dataloader, val_dataloader, tokenizer, epochs=3):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_dataloader, desc='Training'):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            train_steps += 1
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        avg_train_loss = train_loss / train_steps
        print(f'Average training loss: {avg_train_loss}')
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Validation'):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        print(f'Average validation loss: {avg_val_loss}')
    
    return model

# Main function
def main():
    # Load data
    df_train = pd.read_csv("../twitter_training.csv", header=None, names=["id", "category", "sentiment", "tweet_text"])
    df_val = pd.read_csv("../twitter_validation.csv", header=None, names=["id", "category", "sentiment", "tweet_text"])
    df_combined = pd.concat([df_train, df_val], ignore_index=True)
    
    # Map sentiment labels to integers
    sentiment_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
    df_combined['label'] = df_combined['sentiment'].map(sentiment_map)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_combined['tweet_text'].values,
        df_combined['label'].values,
        test_size=0.2,
        random_state=42
    )
    
    # Load tokenizer and model
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3  # Positive, Negative, Neutral
    )
    
    # Create datasets
    train_dataset = TwitterSentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = TwitterSentimentDataset(val_texts, val_labels, tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # Train model
    trained_model = train_model(model, train_dataloader, val_dataloader, tokenizer)
    
    # Save model
    trained_model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    
    print("Model training complete and saved!")

if __name__ == "__main__":
    main() 