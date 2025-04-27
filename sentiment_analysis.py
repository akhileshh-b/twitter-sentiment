import pandas as pd
import os

# Define the path to the CSV file where the data will be stored
csv_file_path = 'filtered_tweets.csv'

# Function to load or create a DataFrame from the existing CSV
def load_or_create_csv():
    # Check if the file already exists
    if os.path.exists(csv_file_path):
        # Load the existing data from the CSV file
        return pd.read_csv(csv_file_path)
    else:
        # If the file doesn't exist, create a new empty DataFrame with columns
        return pd.DataFrame(columns=['id', 'category', 'sentiment', 'tweet_text'])

# Load the dataset without headers (manually assign column names)
column_names = ['id', 'category', 'sentiment', 'tweet_text']
df_new_data = pd.read_csv('twitter_training.csv', header=None, names=column_names, encoding='latin1')

# Function to filter tweets based on user query
def filter_tweets(query, dataframe):
    return dataframe[dataframe['tweet_text'].str.contains(query, case=False, na=False)]

# Main function to interact with the user and append filtered tweets to the CSV
def main():
    # Load or create the CSV file
    df_existing = load_or_create_csv()

    query = input("Enter a hashtag or query (e.g., #AI, Machine Learning): ")
    tweet_count = int(input("Enter the number of tweets you want to fetch (e.g., 100): "))
    
    # Filter tweets based on the query
    filtered_tweets = filter_tweets(query, df_new_data)
    
    # Check if there are enough matching tweets
    if len(filtered_tweets) == 0:
        print(f"No tweets found for query '{query}'.")
        return
    
    # Append the filtered tweets to the existing DataFrame
    df_existing = pd.concat([df_existing, filtered_tweets.head(tweet_count)], ignore_index=True)
    
    # Save the updated DataFrame to the same CSV file
    df_existing.to_csv(csv_file_path, index=False)
    
    print(f"Filtered tweets saved and updated in '{csv_file_path}'")

if __name__ == '__main__':
    main()
