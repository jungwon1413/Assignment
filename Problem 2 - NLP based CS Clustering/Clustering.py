import pandas as pd
import nltk

# Sentiment analyzing module for sentence
from nltk.sentiment.vader import SentimentIntensityAnalyzer    # Need to Install twython to avoid warning
from tqdm import tqdm_notebook
from datetime import datetime
from matplotlib import pyplot as plt

# Load data
tweets = pd.read_csv('Dataset/twcs.csv')

# Get customer requests and company responses 
# Filtering first inbounds
first_inbound = tweets[pd.isnull(tweets.in_response_to_tweet_id) & tweets.inbound]

inbounds_and_outbounds = pd.merge(first_inbound, tweets, left_on='tweet_id', 
                                  right_on='in_response_to_tweet_id')

inbounds_and_outbounds = inbounds_and_outbounds[inbounds_and_outbounds.inbound_y ^ True]

# Enable progress reporting on `df.apply` calls
tqdm_notebook().pandas()

# Instantiate sentiment analyzer from NLTK, make helper function
sentiment_analyzer = SentimentIntensityAnalyzer()

def sentiment_for(text: str) -> float:    # Input type is string, and output type is float
    return sentiment_analyzer.polarity_scores(text)['compound']

# Function test
print("Sentiment Score for 'I love it!': {}".format(sentiment_for('I love it!')))
print("Sentiment Score for 'I would do better than that.': {}".format(sentiment_for('I would do better than that.')))

# Analyze sentiment of inbound customer support requests
inbounds_and_outbounds['inbound_sentiment'] = \
    inbounds_and_outbounds.text_x.progress_apply(sentiment_for)
    
# Details of dataset
inbounds_and_outbounds.head()

# Type of features
print("Dimensions of dataset: inbounds_and_outbounds.shape")
inbounds_and_outbounds.dtypes

# Details of numeric features
inbounds_and_outbounds.describe()

author_grouped = inbounds_and_outbounds.groupby('author_id_y')
top_support_providers = set(author_grouped.agg('count')
                                .sort_values(['tweet_id_x'], ascending=[0])
                                .index[:20]
                                .values)

satisfied_lower = inbounds_and_outbounds['inbound_sentiment'] > 0
satisfied = inbounds_and_outbounds[satisfied_lower]
dissatisfied_upper = inbounds_and_outbounds['inbound_sentiment'] <= 0
dissatisfied = inbounds_and_outbounds[dissatisfied_upper]

inbounds_and_outbounds \
    .loc[inbounds_and_outbounds.author_id_y.isin(top_support_providers)] \
    .groupby('author_id_y') \
    .tweet_id_x.count() \
    .sort_values() \
    .plot('barh', title='Top 20 Brands by Volume')
    
inbounds_and_outbounds \
    .loc[inbounds_and_outbounds.author_id_y.isin(top_support_providers)] \
    .groupby('author_id_y') \
    .inbound_sentiment.mean() \
    .sort_values() \
    .plot('barh', title='Customer Sentiment by Brand (top 20)')
    
satisfied \
    .loc[inbounds_and_outbounds.author_id_y.isin(top_support_providers)] \
    .groupby('author_id_y') \
    .inbound_sentiment.mean() \
    .sort_values() \
    .plot('barh', title='Average Score of Satisfied Customers by Brand (top 20)')
    
dissatisfied \
    .loc[inbounds_and_outbounds.author_id_y.isin(top_support_providers)] \
    .groupby('author_id_y') \
    .inbound_sentiment.mean() \
    .sort_values(ascending=False) \
    .plot('barh', title='Average Score of Dissatisfied Customers by Brand (top 20)')