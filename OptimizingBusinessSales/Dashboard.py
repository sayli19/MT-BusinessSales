import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import string
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
import pandas as pd
import random
import string
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.probability import FreqDist
from SentimentalAnalysis import calculateSentiment
from nltk.sentiment import SentimentIntensityAnalyzer
reviewsCSV = pd.read_csv(
    'E:/Master-Thesis/OptimizingBusinessSales/dataset/ProductReviews-MT.csv')


def fetchStarPercentage(itemID):
    totalReviews = reviewsCSV.loc[reviewsCSV['ID'] == itemID[0]]
    totalReviews = totalReviews[['ID', 'Rating']]

    totalRatings = totalReviews.count().Rating
    oneStar = totalReviews.loc[totalReviews['Rating'] == 1]
    twoStar = totalReviews.loc[totalReviews['Rating'] == 2]
    threeStar = totalReviews.loc[totalReviews['Rating'] == 3]
    fourStar = totalReviews.loc[totalReviews['Rating'] == 4]
    fiveStar = totalReviews.loc[totalReviews['Rating'] == 5]
    arrStar = [format((((oneStar.count().Rating)/totalRatings)*100), ".2f"),
               format((((twoStar.count().Rating)/totalRatings)*100), ".2f"),
               format((((threeStar.count().Rating)/totalRatings)*100), ".2f"),
               format((((fourStar.count().Rating)/totalRatings)*100), ".2f"),
               format((((fiveStar.count().Rating)/totalRatings)*100), ".2f")]
    return arrStar


def fetchEmotionPer(posCount, negCount, neuCount):
    total = posCount + negCount + neuCount
    arrEmo = [format((((posCount)/total)*100), ".2f"),
              format((((negCount)/total)*100), ".2f"),
              format((((neuCount)/total)*100), ".2f")]
    return arrEmo


def fetchCommonWords(itemID, itemName):
    combinedReviews = reviewsCSV.loc[reviewsCSV['ID'] == itemID[0]]
    combinedReviews = combinedReviews[['Reviews', 'Tag']]
    combinedReviews['allReviews'] = combinedReviews['Reviews'] + \
        " " + combinedReviews['Tag']

    combinedReviews = combinedReviews['allReviews']
    reviewLowered = [str(word).lower() for word in combinedReviews]

    title = itemName
    title = [word.lower() for word in title]
    title = ' '.join([str(elem) for elem in title])

    # using list comprehension
    reviewListToString = ' '.join([str(elem) for elem in reviewLowered])
    sia = SentimentIntensityAnalyzer()

    # calculateSentiment(reviewListToString)
    # bow
    stopwords = ['star', 'stars', 'and', 'it',
                 'the', 'these', 'do', 'of', 'but', 'as', 'having', 'seem', 'seems', 'about', 'a', 'actually', 'almost',
                 'also', 'am', 'an', 'are', 'by', 'did', 'do', 'either', 'else', 'for', 'from', 'had', 'has', 'have', 'hence', 'how',
                 'i', 'if', 'in', 'is', 'its', 'just', 'may', 'maybe', 'me', 'might', 'mine', 'must', 'my', 'neither',
                 'nor', 'not', 'of', 'oh', 'ok', 'okay', 'when', 'where', 'were', 'was', 'which', 'while', 'who',
                 'whose', 'why', 'will', 'with', 'yes', 'you', 'your', 'usually', 'even', 'them', 'to', 'off', 'this', 'or', 'that',
                 'on', 'so', 'be', 'we', 'no', 'what', 'than', 'would', 'our', 'all', 'can', 'only', 'does', 'into', 'could',
                 'been', 'then', 'too', "it's", "i'm", 'there', 'still', 'around', "i've", 'every', 'inside', 'outside', 'up', 'down',
                 'go', 'say', 'saying', 'see', 'before', 'after', 'his', 'her', 'onto', 'often', 'myself', "they're", "i'll",
                 "there's", 'and', 'etc', 'although', 'anymore', 'below', 'i', 'rather', 'under', 'told', "i'd", 'doing', 'to',
                 'she', 'he', 'goes', 'also', 'each', 'anyway', 'wife', 'husband', 'they', 'at', 'e.g', ':', '-', '', 'well', "don't", 'now', 'without', 'me', 'more', 'in', 'this', 'made', "e.g"]

    stop = set((string.punctuation))

    # tokenize the data
    tokenizer = WhitespaceTokenizer()

    filteredList = [w for w in tokenizer.tokenize(
        reviewListToString) if w not in (stop and stopwords)]

    # remove period
    periodList = [w.strip(string.punctuation) for w in filteredList]
    title = title.split(" ")
    # most common words
    commonWords = nltk.FreqDist(periodList)
    arrCommonWords = commonWords.most_common(10)

    blanklist = ['', 'very', 'via', ]
    blackListedArra = np.concatenate((blanklist, title))

    arrCommonWords = [
        w for w in arrCommonWords if w[0] not in blackListedArra]
    return arrCommonWords
