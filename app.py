import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")

# Load and combine both datasets
@st.cache_data
def load_data():
    df_train = pd.read_csv("twitter_training.csv", header=None, names=["id", "category", "sentiment", "tweet_text"])
    df_val = pd.read_csv("twitter_validation.csv", header=None, names=["id", "category", "sentiment", "tweet_text"])
    df_combined = pd.concat([df_train, df_val], ignore_index=True)
    df_combined = df_combined.rename(columns={'tweet_text': 'text'})
    return df_combined

df = load_data()

# Header
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>📊 Sentiment Analysis on Twitter Data</h1>
    <p style='text-align: center; font-size:18px;'>Search for any hashtag or keyword to view sentiment insights</p>
    <hr style='border: 1px solid #4CAF50;'>
""", unsafe_allow_html=True)

# Input box
query = st.text_input("Enter a hashtag or keyword to search:", placeholder="Enter your hashtag or query")

# Process and show results
if query:
    filtered = df[df['text'].str.contains(query, case=False, na=False)].copy()

    if not filtered.empty:
        sentiment_counts = filtered['sentiment'].value_counts()
        sentiments = sentiment_counts.index.tolist()
        counts = sentiment_counts.values.tolist()

        # Layout for charts
        col1, col2 = st.columns(2)

        # Bar chart
        with col1:
            st.subheader("📊 Bar Chart")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            bars = ax1.bar(sentiments, counts, color=['#4CAF50', '#FF9800', '#F44336', '#9E9E9E'])
            for bar, label in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{label} tweets', ha='center', fontsize=10, weight='bold')
            ax1.set_ylabel("Tweet Count")
            ax1.set_xlabel("Sentiment")
            ax1.set_title(f'Sentiment Bar Chart')
            ax1.spines[['right', 'top']].set_visible(False)
            st.pyplot(fig1)

        # Pie chart
        with col2:
            st.subheader("🥧 Pie Chart")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.pie(counts, labels=sentiments, autopct='%1.1f%%',
                    colors=['#4CAF50', '#FF9800', '#F44336', '#9E9E9E'], startangle=140)
            ax2.set_title(f'Sentiment Pie Chart')
            st.pyplot(fig2)

        # Show table
        st.subheader("📄 Filtered Tweets")
        st.dataframe(filtered[['id', 'text', 'sentiment']].reset_index(drop=True), use_container_width=True)

        # Download button
        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Tweets as CSV",
            data=csv,
            file_name=f"{query.strip('#')}_filterd_tweets.csv",
            mime='text/csv'
        )
    else:
        st.warning("No tweets found matching your query.")