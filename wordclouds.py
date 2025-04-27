# wordclouds.py
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tweet_scraper import fetch_tweets  # Import fetch_tweets from tweet_scraper.py

def generate_wordcloud(tweets, sentiment_label):
    text = ' '.join([tweets[i] for i in range(len(tweets)) if sentiment_label[i] == sentiment_label])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment_label} Sentiment')
    plt.show()

if __name__ == "__main__":
    query = input("Enter a search query (e.g., AI): ")
    tweets = fetch_tweets(query)

    sentiment_labels = ['Positive', 'Negative', 'Neutral']  # Make sure you have your sentiment list
    for label in sentiment_labels:
        generate_wordcloud(tweets, label)
