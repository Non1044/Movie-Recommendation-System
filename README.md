# 🎬 Movie Recommendation System

![Movie Recommendation picture](asset\pic1.jpg)

This is a content-based movie recommendation system built using Python and machine learning techniques. It recommends movies based on the textual similarity of movie overviews using TF-IDF and sigmoid kernel.

## 📌 Features

- Uses TF-IDF Vectorizer to convert movie overviews into feature vectors.
- Calculates similarity using the sigmoid kernel method.
- Recommends the top 10 movies most similar to the input title.
- Based on TMDb’s movie and credits datasets.

## 🧠 Technologies Used

- Python
- pandas
- numpy
- scikit-learn (`TfidfVectorizer`, `sigmoid_kernel`)

## 🗂 Dataset

- [`tmdb_5000_movies.csv`](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [`tmdb_5000_credits.csv`](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## 🚀 How It Works

1. Load and merge movie and credit datasets.
2. Clean and preprocess the data by removing irrelevant columns.
3. Use `TfidfVectorizer` to vectorize the `overview` text column.
4. Compute pairwise similarity using `sigmoid_kernel`.
5. Recommend the top 10 similar movies based on the selected title.