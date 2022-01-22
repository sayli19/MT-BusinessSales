import pandas as pd
import random
import string
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.probability import FreqDist


def calculateSentiment(reviewListToString):
    df = pd.read_csv(
        'E:/Master-Thesis/OptimizingBusinessSales/dataset/ProductReviews-MT.csv')

    # compress the table to necessary fields
    df = df[['Rating', 'Reviews', 'Tag']]

    # merge review text and summary and create a new column = TAGS
    df['tags'] = df['Reviews'] + " " + df['Tag']

    pos_reviews = df.loc[df['Rating'] > 3]
    neg_reviews = df.loc[df['Rating'] < 3]

    pos_reviews.insert(2, 'reaction', 'positive')
    neg_reviews.insert(2, 'reaction', 'negative')

    review = pd.concat([pos_reviews, neg_reviews], ignore_index=True)

    # segregate pos and neg
    pos_df = review.loc[review['reaction'] == 'positive']
    pos_list = pos_df['tags'].tolist()

    neg_df = review.loc[review['reaction'] == 'negative']
    neg_list = neg_df['tags'].tolist()

    # lowercase
    pos_list_lowered = [str(word).lower() for word in pos_list]
    neg_list_lowered = [str(word).lower() for word in neg_list]

    # using list comprehension
    pos_list_to_string = ' '.join([str(elem) for elem in pos_list_lowered])
    neg_list_to_string = ' '.join([str(elem) for elem in neg_list_lowered])

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
                 'she', 'he', 'goes', 'also', 'each', 'anyway', 'wife', 'husband', 'they', 'at']

    # bow
    stop = set((string.punctuation))
    # tokenize the data
    tokenizer = WhitespaceTokenizer()

    filtered_pos_list = [w for w in tokenizer.tokenize(
        pos_list_to_string) if w not in (stop and stopwords)]
    filtered_neg_list = [w for w in tokenizer.tokenize(
        neg_list_to_string) if w not in (stop and stopwords)]

    # remove period
    filtered_pos_period = [w.strip(string.punctuation)
                           for w in filtered_pos_list]
    filtered_neg_period = [w.strip(string.punctuation)
                           for w in filtered_neg_list]

    # most common words
    common_pos = nltk.FreqDist(filtered_pos_period)
    common_neg = nltk.FreqDist(filtered_neg_period)

    # print(common_pos.most_common(15))
    # print(common_neg.most_common(10))

    # print(common_neg)

    # dictory func

    def word_features(words):
        return dict([(word, True) for word in words.split()])

    negative_features = [(word_features(f), 'neg')
                         for f in filtered_neg_period]
    positive_features = [(word_features(f), 'pos')
                         for f in filtered_pos_period]

    labeledwords = positive_features + negative_features

    random.shuffle(labeledwords)

    train_set, test_set = labeledwords[300000:], labeledwords[:90000]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    # print(classifier.classify(word_features(reviewListToString)))
    # display the list of necessaray information
    # classifier.show_most_informative_features(10)

    # print(classifier.classify(word_features('I cannot speak for the tweeters as i already had a decent factory component system and just needed to upgrade my 6.5 speakers for more crisp midrange and treble as well as eliminating distortion at high volume and frequencies, for that job these speakers did beautifully. please note that these speakers are not a replacement for subs. if you want a systems that bumps you need at least one DECENT sealed subwoofer that covers all of the base and let your interior speakers cover the midrange and treble. I only have an aftermarket head unit powering these speakers as they dont really need an amp unless your head unit cannot provide ample power While these are a good set for someone wanting a beginners component setup. These left me a bit underwhelmed. The tweeters sound like they have no power behind them. Even hooked up to a decent amplifier. I believe it is the fault of the passive crossovers that came in the kit. By comparison, I have a pair of DS18 1" tweeters in front that sound night and day better than these. They work good for the money I guess tweets are good. The drivers well you cant really turn them up very loud with out a lot of distortion and God help if there is a low bass line. Wish I would have spent a little more. I guess you get what you pay for! I have them hooked up to an amp running about 60 watts to each one. I think one has already blown. Only had it for a week. I think the speakers should only be hooked up to your head unit not an amp.Made my system come alive Good component set. Mid range bass to high tweets all filter to complete my trucks sound. Reasonable price for this product. Fit was good. great Good product! The sound quality is terrible! Appearance is deceiving I like this brand alot but these were not the best ive purchased. Great sound for the price. If you are looking for low mids and crisp highs without breaking your wallet, these are the speakers you want to buy! Sounds good For 40 bucks these sound better then the jvc 6.5 and Rockville 4/6 together I will be buying them again good mod bass Im very impressed by the quality you get for the price that you pay. These components are very good... even if I would have paid the full price of 50 or $60 for them I would have still been satisfied. Picture is component speaker but description is for 8" subwoofer. Confusing, These speaker are great for the price. The sound quality is better than the Infinity speakers I had. If you have a standard stereo these are for you.These are great for the price!!')))
    # print(classifier.classify(word_features('i love this product it so so much')))


# Positive    11164
# Neutral      4308
# Negative     1172
