# tweet_scraper.py
import requests
from bs4 import BeautifulSoup

def fetch_tweets(query):
    # Construct Google search URL
    url = f"https://www.google.com/search?q=site:twitter.com+{query}"

    # Send GET request to fetch page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract tweet-like elements (modify according to actual structure)
    tweets = []
    for tweet_div in soup.find_all('div', {'class': 'BVG0Nb'}):  # This class may change over time
        tweet = tweet_div.get_text()
        tweets.append(tweet)

    return tweets

if __name__ == "__main__":
    query = input("Enter a search query (e.g., AI): ")
    tweets = fetch_tweets(query)

    print(f"Fetched {len(tweets)} tweets for the query '{query}'")
    for tweet in tweets[:5]:  # Print first 5 tweets
        print(tweet)
