import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# ^^^imports^^^

rating_constant = 5

# Read in data
movies =  pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

def clean(title):
    """ 
    Removes all non-alphanumeric characters from a string.
    """
    return re.sub("[^a-zA-Z0-9 ]", "", title)

# Remove all non-alphanumeric characters from the titles
movies["cleaned"] = movies["title"].apply(clean)

def search(term):
    """
    Given a term, searches the database for a movie.
    """

    # Create a term frequency matrix
    vector = TfidfVectorizer(ngram_range=(1,2))
    matrix = vector.fit_transform(movies["cleaned"])

    cleaned = clean(term)

    # Find similarity between search term and each movie in DB
    vec = vector.transform([cleaned])
    sim = cosine_similarity(vec, matrix).flatten()

    # Find and return the movies with the 8 most similar titles
    pos = np.argpartition(sim, -8)[-8:]
    res = movies.iloc[pos][::-1]
    return res

def recommend(id): 
    """
    Given a movieId, recommends a movie that users with similar taste rated highly.
    """

    # Finding users who liked the same movie
    simUsers = ratings[(ratings["movieId"] == id) & (ratings["rating"] >= rating_constant)]["userId"].unique()

    # Find the movies that they also liked
    sim_recs = ratings[(ratings["userId"].isin(simUsers)) & (ratings["rating"] >= rating_constant)]["movieId"]
    sim_recs = sim_recs.value_counts() / len(simUsers)
    sim_recs = sim_recs[sim_recs > .1]

    # Separate the random movies out - i.e. they like some movies because they're good movies, not bc they are similar to our search term
    allUsers = ratings[(ratings["movieId"].isin(sim_recs.index)) & (ratings["rating"] >= rating_constant)]
    all_recs = allUsers["movieId"].value_counts() / len(allUsers["userId"].unique())

    # Create a recommendation score
    rec_perc = pd.concat([sim_recs, all_recs], axis=1)
    rec_perc.columns = ["similar", "all"]
    rec_perc["score"] = rec_perc["similar"] / rec_perc["all"]

    # Return the 10 movies with the highest scores
    rec_perc = rec_perc.sort_values("score", ascending=False)
    recs = rec_perc.head(10).merge(movies, left_index=True, right_on="movieId")
    return recs["title"]

# Search for a movie
# search_term = input("Hello, what movie would you like to search for?\n")
# print("Here are the results:")
# print(search(search_term)["title"])

# Recommend a movie
rec_term = input("Hello, what movie do you want to base your recommendations off of?\n")
print("Here are the results:")
movie = search(rec_term)                    # Search for the given movie
id = int((movie.iloc[0])["movieId"])        # Get its movieId
print(recommend(id))