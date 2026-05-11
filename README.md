# Movie Recommendation System

## Overview

This project implements an item-based collaborative filtering recommendation system using the MovieLens dataset.

It generates personalized movie recommendations based on user preferences and similarity between items.

## Features

* Item-based collaborative filtering
* Cosine similarity for recommendation
* User bias normalization (mean-centering)
* RMSE evaluation (~0.96)
* Automatic dataset download

## Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run:
   python recommender.py

## Example Output

Recommendations for: Star Wars (1977)

* Return of the Jedi (1983)
* Empire Strikes Back (1980)
* Raiders of the Lost Ark (1981)

RMSE: ~0.96

## Key Concepts

* Collaborative Filtering
* Cosine Similarity
* User Bias Normalization
* Recommendation Systems

## Future Improvements

* Matrix Factorization (SVD)
* Hybrid recommendation system
* Web interface

## Author

Imaya
