import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the trained model pipeline
@st.cache_resource
def load_model():
    return joblib.load("model_pipeline.pkl")

model = load_model()

st.title("Sentiment Analyzer")

menu = ["Home", "Predict Sentiment"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Project Summary")
    st.write("""
        This application analyzes ChatGPT user reviews and predicts sentiment using NLP and machine learning.
        Navigate to 'Predict Sentiment' to try it out, or explore insights in the 'Insights' tab.
    """)

elif choice == "Predict Sentiment":
    st.subheader("Predict User Sentiment")
    review = st.text_area("Enter a user review below:")
    
    if st.button("Predict"):
        if review.strip():
            prediction = model.predict([review])[0]
            st.success(f"*Predicted Sentiment:* {prediction}")
        else:
            st.warning("Please enter a review before clicking Predict.")

elif choice == "Insights":
    st.subheader("Exploratory Data Analysis (EDA) Insights")
    
    try:
        df = pd.read_csv("chatgpt_reviews - chatgpt_reviews.csv")
    except FileNotFoundError:
        st.error("The file 'reviews.csv' was not found.")
    else:
        df.dropna(subset=["review", "rating", "date"], inplace=True)
        df["sentiment"] = df["rating"].apply(lambda r: "Negative" if r <= 2 else "Neutral" if r == 3 else "Positive")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Sentiment Distribution")
            sentiment_counts = df["sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

        with col2:
            st.write("### Average Rating Over Time")
            df["date"] = pd.to_datetime(df["date"], errors='coerce')
            df.dropna(subset=["date"], inplace=True)
            avg_rating = df.groupby(df["date"].dt.to_period("M"))["rating"].mean()
            avg_rating.index = avg_rating.index.to_timestamp()
            st.line_chart(avg_rating)

        st.write("### Word Cloud by Sentiment")
        sentiment_type = st.selectbox("Choose Sentiment", ["Positive", "Negative"])
        text = " ".join(df[df["sentiment"] == sentiment_type]["review"].dropna())
        if text:
            wordcloud = WordCloud(max_words=100, background_color="white").generate(text)
            st.image(wordcloud.to_array())
        else:
            st.warning(f"No text available for {sentiment_type} sentiment.")