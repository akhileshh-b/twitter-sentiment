# Deep Learning Based Twitter Sentiment Analysis

This project uses deep learning models to analyze the sentiment of tweets based on a search query. The application:

1. Uses pre-trained transformer models (BERT/RoBERTa) for sentiment analysis
2. Processes tweets and classifies them as positive, negative, or neutral
3. Visualizes sentiment distribution with charts
4. Generates word clouds for different sentiment categories

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Model Architecture

The sentiment analysis is performed using a pre-trained transformer model (BERT/RoBERTa) fine-tuned on sentiment classification tasks. The model:

1. Tokenizes input text
2. Generates contextual embeddings
3. Classifies sentiment using a classification head

## Data Processing

The application processes Twitter data by:
1. Loading tweets from CSV files
2. Preprocessing text (tokenization, etc.)
3. Passing through the transformer model
4. Aggregating sentiment predictions 