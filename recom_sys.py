import pandas as pd
import numpy as np
import requests
import zipfile
import io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
response = requests.get(url)

z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall()

ratings = pd.read_csv(
    "ml-100k/u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    "ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    usecols=[0, 1],
    names=["item_id", "title"]
)

df = pd.merge(ratings, movies, on="item_id")

movie_counts = df['title'].value_counts()
popular_movies = movie_counts[movie_counts > 50].index
df = df[df['title'].isin(popular_movies)]

user_item_matrix = df.pivot_table(
    index="user_id",
    columns="title",
    values="rating"
)

user_item_matrix_norm = user_item_matrix.subtract(user_item_matrix.mean(axis=1), axis=0)
user_item_matrix_filled = user_item_matrix_norm.fillna(0)

item_similarity = cosine_similarity(user_item_matrix_filled.T)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

def recommend_movies(movie_name, top_n=5):
    if movie_name not in item_similarity_df.columns:
        return "Movie not found"
    
    similar_scores = item_similarity_df[movie_name]
    similar_scores = similar_scores.sort_values(ascending=False)
    
    return similar_scores.iloc[1:top_n+1]

def calculate_rmse():
    R = user_item_matrix.values
    R_filled = user_item_matrix.fillna(0).values

    user_mean = user_item_matrix.mean(axis=1).values.reshape(-1, 1)
    R_centered = R_filled - user_mean

    pred = np.zeros(R.shape)

    for i in range(R.shape[1]):
        sim_scores = item_similarity[i].copy()
        sim_scores[i] = 0

        numerator = R_centered.dot(sim_scores)
        denominator = np.abs(sim_scores).sum()

        if denominator == 0:
            denominator = 1

        pred[:, i] = user_mean.flatten() + (numerator / denominator)

    mask = ~np.isnan(R)

    return np.sqrt(mean_squared_error(R[mask], pred[mask]))
if __name__ == "__main__":
    
    movie = "Star Wars (1977)"
    
    print("Recommendations for:", movie)
    print(recommend_movies(movie))
    
    print("\nRMSE:", calculate_rmse())
