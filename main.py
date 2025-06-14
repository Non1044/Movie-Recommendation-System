import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel


credits = pd.read_csv("tmdb_5000_credits.csv")
movies = pd.read_csv("tmdb_5000_movies.csv")


def main():
    credits.index = credits.index.astype(str)
    credits_column_renamed = credits.rename(columns={'movie_id': "id"})
    movies_merge = movies.merge(credits_column_renamed)
    movies_cleaned = movies_merge.drop(
        columns=['homepage', 'status', 'production_countries'])

    tfv = TfidfVectorizer(min_df=3,  max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3),
                          stop_words='english')
    movies_cleaned_df = movies_cleaned[['overview']].fillna('')
    tfv_matrix = tfv.fit_transform(movies_cleaned_df['overview'])
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    indices = pd.Series(
        movies_cleaned.index, index=movies_cleaned['original_title']).drop_duplicates()

    def give_recomendations(title, sig=sig):
        # Get the index corresponding to original_title
        idx = indices[title]
        # Get the pairwsie similarity scores
        sig_scores = list(enumerate(sig[idx]))
        # Sort the movies
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
        # Scores of the 10 most similar movies
        sig_scores = sig_scores[1:11]
        # Movie indices
        movie_indices = [i[0] for i in sig_scores]
        # Top 10 most similar movies
        return movies_cleaned['original_title'].iloc[movie_indices]

    result = give_recomendations('The Avengers')
    print(result)


if __name__ == "__main__":
    main()
