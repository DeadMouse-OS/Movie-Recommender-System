import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import pickle

movies_data = pd.read_csv("/content/tmdb_5000_movies.csv")
credits_data = pd.read_csv("/content/tmdb_5000_credits.csv")

movies_data = movies_data.merge(credits_data, on = "title")

movies_data.info()

good_cols = ['id' , 'title', 'genres', 'overview', 'keywords', 'cast', 'crew']
num_good_cols = ['release_date', 'revenue']
movie_data = movies_data[good_cols].copy()
movie_data = movie_data.dropna()


import ast
def helper(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def helper_cast(obj):
    L=[]
    f=0
    for i in ast.literal_eval(obj):
      if(f!=3):
        L.append(i['name'])
        f =f+1
      else:
        break
    return L

def fetch_names(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if(i['job'] == 'Director'):
            L.append(i['name'])
            break
    return L

def return_names(obj):
  return " ".join(obj)

movie_data["genres"] = movie_data["genres"].apply(helper)
movie_data["keywords"] = movie_data["keywords"].apply(helper)
movie_data["crew"] = movie_data['crew'].apply(fetch_names)
movie_data["cast"] = movie_data["cast"].apply(helper_cast)
movie_data["overview"] = movie_data['overview'].apply(lambda x: x.split())
movie_data["title"] = movie_data['title'].apply(lambda x: x.split())

movie_data["genres"] = movie_data['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movie_data["keywords"] = movie_data['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movie_data["cast"] = movie_data['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movie_data["overview"] = movie_data['overview'].apply(lambda x: [i.replace(" ", "") for i in x])
movie_data["crew"] = movie_data['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movie_data["tags"] = movie_data['title'] + movie_data["genres"]  + movie_data["keywords"]  + movie_data["cast"]  + movie_data["crew"]

movie_data["title"] = movie_data["title"].apply(lambda x: " ".join(x))

df = movie_data[['id','title', 'tags']]
df["tags"] = df["tags"].apply(lambda x: " ".join(x))

cv = CountVectorizer(max_features = 5000, stop_words = 'english')
vectors = cv.fit_transform(df['tags']).toarray()

ps = PorterStemmer()
def stem_text(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

df['tags'] = df['tags'].apply(stem_text)

sim = cosine_similarity(vectors)

def recommend(movie):
    index = movie_data[movie_data["title"] == movie].index[0]
    sm = sim[index]
    m_list = sorted(list(enumerate(sm)), reverse = True, key = lambda x: x[1])[1:6]

    for i in m_list():
        print(df.iloc[i[0]].title)

pickle.dump(df, open("movies.pkl", 'wb'))
pickle.dump(df.to_dict(), open("movie_dict.pkl", 'wb'))
pickle.dump(sim, open("similarity.pkl", 'wb'))