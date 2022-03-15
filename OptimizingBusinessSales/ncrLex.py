from nrclex import NRCLex
import pandas as pd
import random
import string
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.probability import FreqDist
import plotly.express as px

df = pd.read_csv(
    "E:/Master-Thesis/OptimizingBusinessSales/dataset/SubPol.csv")
df = df['tags']
stringData = df.str.cat(sep=', ')

text_object = NRCLex(stringData)
data = text_object.raw_emotion_scores

emotion_df = pd.DataFrame.from_dict(data, orient='index')
emotion_df = emotion_df.reset_index()
emotion_df = emotion_df.rename(
    columns={'index': 'Emotion Classification', 0: 'Emotion Count'})
emotion_df = emotion_df.sort_values(by=['Emotion Count'], ascending=False)
fig = px.bar(emotion_df, x='Emotion Classification', y='Emotion Count',
             color='Emotion Classification', orientation='v', width=1000, height=800)
fig.show()
