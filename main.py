import streamlit as st;
import pandas as pd;
import plotly.express as px;
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.title(':red[MOVIE RECOMMENDATIONS]');
title_pg = st.selectbox("Select your", ('popularity', 'similarity'))
if title_pg == 'similarity':

    st.title(":red[SIMILARITY] BASED")
    movies = pd.read_csv('movies.csv')
    tfidf = TfidfVectorizer(stop_words='english');
    movies['overview'] = movies['overview'].fillna(" ")
    matrix = tfidf.fit_transform(movies["overview"])
    pd.DataFrame(matrix.toarray(), columns=tfidf.get_feature_names_out())
    similarity_matrix = linear_kernel(matrix, matrix)
    st.info("Movie Database")
    d = movies['title']
    st.write(d)

    def similarity(title, nr):
        idx = movies.index[movies["title"] == title].tolist()[0];
        scores = list(enumerate(similarity_matrix[idx]));
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        movie_index = [tpl[0] for tpl in scores[1:nr + 1]]
        sim = list(movies["title"].iloc[movie_index])
        return sim

    movie = st.text_input("Enter the movies name", value="Avatar")
    # m = movie.title()
    number = st.slider("Enter the number of similar movies you want:", min_value=1, max_value=30)
    result = similarity(movie, number)
    x = len(result)
    st.subheader(f"the {x} similar movies to {movie}:-")
    for r in result:
        st.write(f"{r}")

if title_pg == "popularity":
    st.title(':red[POPULARITY] BASED')

    movies = pd.read_csv('movies.csv');

    m = movies['vote_count'].quantile(0.8);
    c = movies['vote_average'].mean()
    movies_filtered = movies.loc[movies['vote_count'] > m]
    y = st.slider("top?", min_value=1, max_value=len(movies_filtered))

    def weigthed(df, m=m, c=c):
        r = df['vote_average']
        v = df['vote_count']
        wr = (v / (v + m)) * r + (m / (v + m) * c);
        return wr;

    movies_filtered['weighted'] = movies_filtered.apply(weigthed, axis=1);

    x = movies_filtered.sort_values('weighted', ascending=False)[['original_title', 'weighted']].head(y)
    st.info(f"Top {y} movies")
    st.write(x)
    z = movies_filtered.sort_values('weighted', ascending=False)[['original_title', 'release_date']].head(y)
    st.title(f"the {y} most popular movies")
    fig = px.scatter(movies_filtered, x=x['original_title'], y=x['weighted']);
    st.plotly_chart(fig)
    st.title("They were released on");
    fig1 = px.line(movies_filtered, x=z['original_title'], y=z['release_date']);
    st.plotly_chart(fig1)
