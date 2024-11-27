import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate


# Load movies and ratings data
@st.cache_data
def load_data():
    movies = pd.read_csv(r"D:\Data Science Projects\movie recommendation\dataset\ml-latest-small\movies.csv", encoding="utf-8")
    ratings =pd.read_csv(r"D:\Data Science Projects\movie recommendation\dataset\ml-latest-small\ratings.csv", encoding="utf-8")
    return movies, ratings



movies, ratings = load_data()

st.title("Movie Recommendation System")
st.write("Exploring the MovieLens Dataset")

# Preview datasets
if st.checkbox("Show Movies Dataset"):
    st.write(movies.head())

if st.checkbox("Show Ratings Dataset"):
    st.write(ratings.head())


# Generate Content-Based Recommendations
@st.cache_data
def content_based_recommender(movie_title, movies_df, top_n=10):
    # Vectorize genres
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["genres"].fillna(""))

    # Compute similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get index of the movie
    indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()
    idx = indices[movie_title]

    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    # Get recommended movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return recommended movies
    return movies_df["title"].iloc[movie_indices]

# User Input
movie_title = st.text_input("Enter a movie you like:", "Toy Story (1995)")
if movie_title:
    try:
        recommendations = content_based_recommender(movie_title, movies)
        st.subheader("Content-Based Recommendations:")
        for i, movie in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie}")
    except KeyError:
        st.error("Movie not found. Please try another title.")


# Collaborative Filtering using Surprise
@st.cache_data
def collaborative_recommender(user_id, ratings_df, top_n=10):
    # Prepare data for Surprise
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)

    # Train model
    model = SVD()
    cross_validate(model, data, cv=5, verbose=False)

    # Train on full data
    trainset = data.build_full_trainset()
    model.fit(trainset)

    # Predict ratings for all movies not rated by the user
    movie_ids = ratings_df["movieId"].unique()
    rated_movies = ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist()
    unrated_movies = [movie for movie in movie_ids if movie not in rated_movies]

    predictions = [
        (movie, model.predict(user_id, movie).est) for movie in unrated_movies
    ]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Get top-n recommendations
    top_recommendations = predictions[:top_n]
    recommended_movie_ids = [rec[0] for rec in top_recommendations]
    return movies[movies["movieId"].isin(recommended_movie_ids)]["title"]

# User Input
user_id = st.number_input("Enter your User ID:", min_value=1, max_value=int(ratings["userId"].max()))
if st.button("Get Collaborative Recommendations"):
    recommendations = collaborative_recommender(user_id, ratings)
    st.subheader("Collaborative Filtering Recommendations:")
    for i, movie in enumerate(recommendations, start=1):
        st.write(f"{i}. {movie}")
