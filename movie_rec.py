import pandas as pd
import re

movies =  pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

def clean(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

movies["cleaned"] = movies["title"].apply(clean)

print(movies["title"][0])
print(movies["cleaned"][0])
