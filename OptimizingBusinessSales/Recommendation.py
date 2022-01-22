from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

recommended_products = []


allProducts = pd.read_csv(
    'E:/Master-Thesis/OptimizingBusinessSales/dataset/ProductDetails-MT.csv')
# allProducts = allProducts[['ID', 'name', 'Description', 'Tags']]
allProducts['tags'] = allProducts['name'] + \
    allProducts['Description'] + allProducts['Tags']
allProducts = allProducts.drop(columns=['Description', 'Tags'])

# cv = CountVectorizer(max_features=5000, stop_words='english')
# vector = cv.fit_transform(allProducts['tags']).toarray()
# similarity = cosine_similarity(vector)

# print(allProducts[allProducts['name'] == 'Sanus TV Motion Mount'])


# def recommend(recProduct):
#     index = allProducts[allProducts['name'] == recProduct].index[0]
#     distances = sorted(
#         list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
#     for i in distances[1:6]:
#         print(allProducts.iloc[i[0]].name)


# recommend('DENAQ - AC Adapter')


# new method
tfidf = TfidfVectorizer(stop_words='english')
allProducts['tags'] = allProducts['tags'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(allProducts['tags'])


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(allProducts.index,
                    index=allProducts['name']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies

    return allProducts.iloc[movie_indices]
