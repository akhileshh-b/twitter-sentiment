import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Define the path to the CSV file containing the tweets
csv_file_path = 'filtered_tweets.csv'

# Load the existing CSV containing tweets
df_existing = pd.read_csv(csv_file_path)

# Perform sentiment analysis using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    # Classify sentiment as positive, negative, or neutral based on polarity
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Perform sentiment analysis on the tweet text if sentiment is not already assigned
if 'sentiment' not in df_existing.columns:
    df_existing['sentiment'] = df_existing['tweet_text'].apply(get_sentiment)

# Visualize sentiment distribution using a bar chart and pie chart
def visualize_sentiment(df):
    sentiment_counts = df['sentiment'].value_counts()
    
    # Bar Chart
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
    plt.title('Sentiment Distribution of Tweets')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    # Pie Chart with Percentages
    plt.figure(figsize=(8, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99', '#ff6666'])
    plt.title('Sentiment Distribution (Pie Chart)')
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular
    plt.show()

# Visualize sentiment
visualize_sentiment(df_existing)
