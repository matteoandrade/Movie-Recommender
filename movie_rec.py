import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def clean(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

movies =  pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

movies["cleaned"] = movies["title"].apply(clean)

vector = TfidfVectorizer(ngram_range=(1,2))
matrix = vector.fit_transform(movies["cleaned"])

def search(term):
    cleaned = clean(term)
    vec = vector.transform([cleaned])
    sim = cosine_similarity(vec, matrix).flatten()
    pos = np.argpartition(sim, -8)[-8:]
    res = movies.iloc[pos][::-1]
    return res

def recommend(term):
    id = 1
    simUsers = ratings[(ratings["movieId"] == id) & (ratings["rating"] >= 5)]["userId"].unique()
    sim_recs = ratings[(ratings["userId"].isin(simUsers)) & (ratings["rating"] > 4)]["movieId"]
    sim_recs = sim_recs.value_counts() / len(simUsers)
    sim_recs = sim_recs[sim_recs > .1]
    allUsers = ratings[(ratings["movieId"].isin(sim_recs.index)) & (ratings["rating"] > 4)]
    all_recs = allUsers["movieId"].value_counts() / len(allUsers["userId"].unique())
    rec_perc = pd.concat([sim_recs, all_recs], axis=1)
    rec_perc.columns = ["similar", "all"]
    rec_perc["score"] = rec_perc["similar"] / rec_perc["all"]
    rec_perc = rec_perc.sort_values("score", ascending=False)
    recs = rec_perc.head(10).merge(movies, left_index=True, right_on="movieId")
    print("\nREC\n")
    print(recs["title"])

recommend("hi")

search_term = input("Hello, what movie would you like to search for?\n")
print("Here are the results:")
print(search(search_term)["title"])
