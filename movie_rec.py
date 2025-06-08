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

search_term = input("Hello, what movie would you like to search for?\n")
print("Here are the results:")
print(search(search_term)["title"])
