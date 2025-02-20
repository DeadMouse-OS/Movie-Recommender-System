import streamlit as st
import pickle
import pandas as pd
import requests

# Load movie data and similarity matrix

with open(r"C:\Users\mgura\PycharmProjects\PythonProject\ML_Proj\Movie-Recommender-System\.venv\movie_dict.pkl", 'rb') as file:
    movies_dict = pickle.load(file)

with open(r"C:\Users\mgura\PycharmProjects\PythonProject\ML_Proj\Movie-Recommender-System\.venv\similarity.pkl", 'rb') as file:
    similarity = pickle.load(file)

movies = pd.DataFrame(movies_dict)

# Function to fetch movie poster
def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=1264fd367ea0212e752f166994f0a03f')
    data = response.json()
    return "http://image.tmdb.org/t/p/w500/" + data['poster_path']

# Function to recommend movies
def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    sm = similarity[index]
    m_list = sorted(list(enumerate(sm)), reverse=True, key=lambda x: x[1])[1:6]

    recommended = []
    posters = []
    for i in m_list:
        movie_id = movies.iloc[i[0]].id
        recommended.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))
    return recommended, posters

# Streamlit app interface
st.title("Movie Recommender System")

movie_name = st.selectbox('Select a movie', movies['title'].values)

if st.button("Recommend"):
    names, posters = recommend(movie_name)
    # for name in names:
    #    st.write(name)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])


