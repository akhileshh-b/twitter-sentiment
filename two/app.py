import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm

# Page config
st.set_page_config(page_title="Deep Learning Twitter Sentiment Analysis", layout="wide")

# Load the pre-trained model and tokenizer
@st.cache_resource
def load_model():
    # Using a pre-trained model fine-tuned for sentiment analysis
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

# Load and combine both datasets
@st.cache_data
def load_data():
    df_train = pd.read_csv("../twitter_training.csv", header=None, names=["id", "category", "sentiment", "tweet_text"])
    df_val = pd.read_csv("../twitter_validation.csv", header=None, names=["id", "category", "sentiment", "tweet_text"])
    df_combined = pd.concat([df_train, df_val], ignore_index=True)
    df_combined = df_combined.rename(columns={'tweet_text': 'text'})
    return df_combined

# Function to predict sentiment using the deep learning model
def predict_sentiment(text, model, tokenizer):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
        predicted_class = torch.argmax(predictions, dim=1).item()
    
    # Map model output to sentiment labels
    # The model outputs 1-5 stars, we map to positive/negative/neutral
    if predicted_class >= 4:
        return "Positive"
    elif predicted_class <= 2:
        return "Negative"
    else:
        return "Neutral"

# Load model and data
model, tokenizer = load_model()
df = load_data()

# Header
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🤖 Deep Learning Sentiment Analysis on Twitter Data</h1>
    <p style='text-align: center; font-size:18px;'>Using BERT transformer model for advanced sentiment analysis</p>
    <hr style='border: 1px solid #4CAF50;'>
""", unsafe_allow_html=True)

# Input box
query = st.text_input("Enter a hashtag or keyword to search:", placeholder="Enter your hashtag or query")

# Process and show results
if query:
    filtered = df[df['text'].str.contains(query, case=False, na=False)].copy()
    
    if not filtered.empty:
        # Show a progress bar for sentiment analysis
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Perform sentiment analysis using the deep learning model
        status_text.text("Analyzing sentiments using BERT model...")
        filtered['dl_sentiment'] = None
        
        for i, row in enumerate(filtered.iterrows()):
            text = row[1]['text']
            filtered.at[row[0], 'dl_sentiment'] = predict_sentiment(text, model, tokenizer)
            progress_bar.progress((i + 1) / len(filtered))
        
        status_text.text("Analysis complete!")
        
        # Get sentiment counts
        sentiment_counts = filtered['dl_sentiment'].value_counts()
        sentiments = sentiment_counts.index.tolist()
        counts = sentiment_counts.values.tolist()
        
        # Layout for charts
        col1, col2 = st.columns(2)
        
        # Bar chart
        with col1:
            st.subheader("📊 Sentiment Distribution (BERT)")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            bars = ax1.bar(sentiments, counts, color=['#4CAF50', '#FF9800', '#F44336'])
            for bar, label in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{label} tweets', ha='center', fontsize=10, weight='bold')
            ax1.set_ylabel("Tweet Count")
            ax1.set_xlabel("Sentiment")
            ax1.set_title(f'Sentiment Distribution (BERT Model)')
            ax1.spines[['right', 'top']].set_visible(False)
            st.pyplot(fig1)
        
        # Pie chart
        with col2:
            st.subheader("🥧 Sentiment Distribution (BERT)")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.pie(counts, labels=sentiments, autopct='%1.1f%%',
                    colors=['#4CAF50', '#FF9800', '#F44336'], startangle=140)
            ax2.set_title(f'Sentiment Distribution (BERT Model)')
            st.pyplot(fig2)
        
        # Show table
        st.subheader("📄 Analyzed Tweets")
        st.dataframe(filtered[['id', 'text', 'dl_sentiment']].reset_index(drop=True), use_container_width=True)
        
        # Download button
        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Analyzed Tweets as CSV",
            data=csv,
            file_name=f"{query.strip('#')}_analyzed_tweets.csv",
            mime='text/csv'
        )
        
        # Model explanation
        st.markdown("""
        ## 🤖 Model Architecture
        
        This sentiment analysis uses a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model:
        
        1. **Tokenization**: Text is split into tokens and converted to embeddings
        2. **Transformer Layers**: 12 layers of bidirectional attention mechanisms
        3. **Classification Head**: Final layer that outputs sentiment probabilities
        
        The model was pre-trained on 3.3 billion words and fine-tuned for sentiment analysis.
        """)
    else:
        st.warning("No tweets found matching your query.") 