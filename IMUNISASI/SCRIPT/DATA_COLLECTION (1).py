#!/usr/bin/env python
# coding: utf-8

# In[18]:


# DATA COLLECTION #
import os
import pandas as pd
import tweepy
import re
import string
from textblob import TextBlob
import preprocessor as p
#from preprocessor import clean, tokenize, parse
import nltk
import emoji

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


#Twitter credentials for the app
consumer_key = "wJvckQyRGO9j6IP1HLEqKHy2T"
consumer_secret = "vgActfGgy0UCGR5OGSGRNPiX75jIeXlNGpTtFsqStVjIY6EiIR"
access_key = "150884743-9FLqDlzvx6FFxZvhms2m4CItsFR2ANcJR4liz1HI"
access_secret = "Epn0ZK4uF13BMEkyIZqk7KToeXkymOl7yVG5gCNutRsFQ"

#pass twitter credentials to tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

#file location changed to "data/telemedicine_data_extraction/" for clearer path
today = f"{datetime.now():%Y-%m-%d}"
imunisasi_tweets = "IMUNISASI/imunisasi_data_"+today+".csv"


#columns of the csv file
COLS = ['id', 'created_at', 'source', 'clean_text', 'sentiment','polarity','subjectivity', 'lang',
        'favorite_count', 'retweet_count', 'original_author', 'possibly_sensitive', 'hashtags',
        'user_mentions', 'place', 'place_coord_boundaries', 'original_text_utf8']

#set two date variables for date range
start_date = (datetime.now() + timedelta(days=-1)).strftime("%Y-%m-%d")
#end_date = '2019-08-18'

# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])

# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])

#Emoji patterns
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

#combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)

#print(emoticons)

#mrhod clean_tweets()
def clean_tweets(tweet):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)

    #after tweepy preprocessing the colon left remain after removing mentions
    #or RT sign in the beginning of the tweet
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    #replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)


    #remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)

    #filter using NLTK library append it to a string
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []

    #looping through conditions
    for w in word_tokens:
        #check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)
    #print(word_tokens)
    #print(filtered_sentence)

#method write_tweets()
def write_tweets(keyword, file):
    # If the file exists, then read the existing data from the CSV file.
    if os.path.exists(file):
        df = pd.read_csv(file, header=0)
    else:
        df = pd.DataFrame(columns=COLS)
    #page attribute in tweepy.cursor and iteration
    for page in tweepy.Cursor(api.search, q=keyword,
                              count=10000, include_rts=False, since=start_date).pages(50):
        for status in page:
            if(status.created_at.date() < datetime.now().date()) :
                new_entry = []
                status = status._json

                ## check whether the tweet is in english or skip to the next tweet
                if status['lang'] != 'in':
                    continue

                #when run the code, below code replaces the retweet amount and
                #no of favorires that are changed since last download.
                if status['created_at'] in df['created_at'].values:
                    i = df.loc[df['created_at'] == status['created_at']].index[0]
                    if status['favorite_count'] != df.at[i, 'favorite_count'] or                        status['retweet_count'] != df.at[i, 'retweet_count']:
                        df.at[i, 'favorite_count'] = status['favorite_count']
                        df.at[i, 'retweet_count'] = status['retweet_count']
                    continue


               #tweepy preprocessing called for basic preprocessing
                #clean_text = p.clean(status['text'])
                clean_text = status['text']

                #call clean_tweet method for extra preprocessing
                filtered_tweet=clean_tweets(clean_text)

                #pass textBlob method for sentiment calculations
                blob = TextBlob(filtered_tweet)
                Sentiment = blob.sentiment

                #seperate polarity and subjectivity in to two variables
                polarity = Sentiment.polarity
                subjectivity = Sentiment.subjectivity

                #new entry append status['text'],
                new_entry += [status['id'], status['created_at'],
                              status['source'], filtered_tweet, Sentiment,polarity,subjectivity, status['lang'],
                              status['favorite_count'], status['retweet_count']]

                #to append original author of the tweet
                new_entry.append(status['user']['screen_name'])

                try:
                    is_sensitive = status['possibly_sensitive']
                except KeyError:
                    is_sensitive = None
                new_entry.append(is_sensitive)

                # hashtagas and mentiones are saved using comma separted
                hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
                new_entry.append(hashtags)
                mentions = ", ".join([mention['screen_name'] for mention in status['entities']['user_mentions']])
                new_entry.append(mentions)

                #get location of the tweet if possible
                try:
                    location = status['user']['location']
                except TypeError:
                    location = ''
                new_entry.append(location)

                try:
                    coordinates = [coord for loc in status['place']['bounding_box']['coordinates'] for coord in loc]
                except TypeError:
                    coordinates = None
                new_entry.append(coordinates)

                #textutf8 = status['text'].encode('utf-8')
                #new_entry.append(textutf8)

                textutf8 = status['text'].encode('utf-8')
                new_entry.append(textutf8)

                #textdemojize = emoji.demojize(status['text'])
                #new_entry.append(textdemojize)

                single_tweet_df = pd.DataFrame([new_entry], columns=COLS)
                df = df.append(single_tweet_df, ignore_index=True)
                csvFile = open(file, 'a' ,encoding='utf-8')
    df.to_csv(csvFile, mode='a', columns=COLS, index=False, encoding="utf-8")

#declare keywords as a query for three categories
imunisasi_keywords = 'imunisasi OR vaksin OR vaksinasi'


#call main method passing keywords and file path
write_tweets(imunisasi_keywords,  imunisasi_tweets)


# In[ ]:





# In[ ]:




