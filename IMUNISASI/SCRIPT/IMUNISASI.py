#!/usr/bin/env python
# coding: utf-8

# In[307]:


#Data collection, save in CSV file
import tweepy
import csv
import time
from datetime import datetime
from datetime import timedelta


consumer_key = "wJvckQyRGO9j6IP1HLEqKHy2T"
consumer_secret = "vgActfGgy0UCGR5OGSGRNPiX75jIeXlNGpTtFsqStVjIY6EiIR"
access_token = "150884743-9FLqDlzvx6FFxZvhms2m4CItsFR2ANcJR4liz1HI"
access_token_secret = "Epn0ZK4uF13BMEkyIZqk7KToeXkymOl7yVG5gCNutRsFQ"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

yesterday = (datetime.now() + timedelta(days=-1)).strftime("%Y-%m-%d")
csvFile = open('IMUNISASI/imunisasi_tweet_'+yesterday+'.csv', 'a')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(["id","created_at","text","time_zone","place","coordinates","location","user_name","screen_name","retweet_count","favourite_count"])

for tweet in tweepy.Cursor(api.search,q="'imunisasi OR vaksin OR vaksinasi' -filter:retweets",count=10000,lang="id",since=yesterday,tweet_mode='extended').items():
                
    if(tweet.created_at.date() < datetime.now().date()) :
        print (tweet.full_text.encode('unicode escape')) 
        csvWriter.writerow([tweet.id,
                            tweet.created_at,
                            tweet.full_text.encode('utf-8'),
                            tweet.user.time_zone,
                            tweet.place, 
                            tweet.coordinates,
                            tweet.user.location.encode('utf-8'),
                            tweet.user.name.encode('utf-8'),
                            tweet.user.screen_name.encode('utf-8'),
                            tweet.retweet_count,
                            tweet.favorite_count
                           ])


# In[27]:


import pandas as pd
import nltk
from nltk.stem.porter import *
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
import requests
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
from googletrans import Translator
translator = Translator()


# In[28]:


#MENAMBHAKAN SENTIMEN VADER
def sentiment_analyzer_scores(text, engl=True):
    if engl:
        trans = text
    else:
        trans = translator.translate(text).text

    score = analyser.polarity_scores(trans)
    lb = score['compound']
    if lb >= 0.05:
        return 1
    elif (lb > -0.05) and (lb < 0.05):
        return 0
    else:
        return -1


# In[29]:


import seaborn as sns
def anl_tweets(lst, title='Tweets Sentiment', engl=False ): #engl = False karena pake bahasa indonesia
    sents = []
    for tw in lst:
        try:
            st = sentiment_analyzer_scores(tw, engl)
            sents.append(st)
        except:
            sents.append(0)
    ax = sns.distplot(
        sents,
        kde=False,
        bins=3)
    ax.set(xlabel='Negative                Neutral                 Positive',
           ylabel='#Tweets',
          title="Tweets of @"+title)
    return sents


# In[206]:


#READ DATA
file_name = 'IMUNISASI/imunisasi_tweet_2019-08-19.csv'
file_name.encode()
df_tws = pd.read_csv(file_name)
df_tws.shape
df_tws.head()


# In[183]:


#ANALYZE: sentimen vader , tambah kolom sent_vader
df_tws['sent_vader'] = anl_tweets(df_tws.text)


# In[313]:


#extract EMOJI
#emoji extract
def split_count(text):
    #text.decode('unicode-escape')
    emoji_list = []
    #data = regex.findall(r'\X', text)
    data = regex.findall(r'[^\x00-\x7F]+',text)
    flags = regex.findall(u'[\U0001F1E6-\U0001F1FF]', text) 
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)

    return emoji_list

line = [b'Abis vaksin anjjjj gabisa tidur madep kiri ya Allah:((((( maen hp aja sakit lengannya \xf0\x9f\x98\xad AUG 27\xf0\x9f\xa6\x8b']
a=str(line[0],'utf-8')
counter = split_count(a)
emoji_all = ','.join(emoji for emoji in counter)
print(emoji_all)
print(emoji_all.encode('utf-8'))


# In[314]:


def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

HT_all = hashtag_extract(df_tws['text'])
#print(HT_all)
HT_all = sum(HT_all,[])
print(HT_all)


# In[311]:


#emoji free text
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
def clean_tweets(tweet):
    
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
    #tweet = re.sub("""[\s"'#]+\w+""",' ', tweet)
    tweet = emoji_pattern.sub(r'', tweet)
    
    # remove twitter Return handles (RT @xxx:)
    tweet = np.vectorize(remove_pattern)(tweet, "RT @[\w]*:")
    # remove URL links (httpxxx)
    tweet = np.vectorize(remove_pattern)(tweet, "https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    tweet = np.core.defchararray.replace(tweet, "[^a-zA-Z#]", " ")
    # remove special characters enter \n
    tweet = np.core.defchararray.replace(tweet, "\n", " ")
    # remove special characters b'
    #tweet = np.core.defchararray.replace(tweet, "b'", "")

    return tweet

line = [b'Abis vaksin anjjjj gabisa tidur madep kiri ya Allah:((((( maen hp aja sakit lengannya \xf0\x9f\x98\xad AUG 27\xf0\x9f\xa6\x8b']
a=str(line[0],'utf-8')
print(clean_tweets(a))


# In[265]:


#DATA CLEANING, tambah kolom text_clean
df_tws['text_clean'] = clean_tweets(df_tws.text)
df_tws.head()


# In[291]:


# collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

HT_all = hashtag_extract(df_tws['text'])
#print(HT_all)
HT_all = sum(HT_all,[])
print(HT_all)

#show bar plot
a = nltk.FreqDist(HT_all)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[23]:


#VISUALISASI
#WORD CLOUD
word_cloud(df_tws.text)
# Words in negative tweets
tws_pos = df_tws['text'][df_tws['sent_vader'] == -1]
word_cloud(tws_pos)


# In[ ]:


#PREPROCESSING
import re
import string
import csv

file_data_normalisasi ="C:/anmedsos/proyek/normalisasi.csv"
file_data_asli ="C:/anmedsos/proyek/data_awal.csv"
file_data_preprocessing ="C:/anmedsos/proyek/data_preprocessing.csv"


from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
stopwords_factory = StopWordRemoverFactory()
stopword = stopwords_factory.create_stop_word_remover()
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
reader = csv.reader(open(file_data_normalisasi, 'r')) #Sumber : Rinaldo (2016)
data_asli = csv.reader(open(file_data_asli, 'r'))
fw = open(file_data_preprocessing, 'a')

d = {}
for row in reader :
    k,v = row
    k = k.lower()
    v = v.lower()
    d[k] = v
pat = re.compile(r"\b(%s)\b" % "|".join(d))
for row in data_asli :
#===================================NORMALISASI KATA===========================
    print("Teks Asli :"+str(row))
    text = str(row).lower()
    text = pat.sub(lambda m: d.get(m.group()), text)
    print("Proses Normalisasi :"+text)
    
    #text = text.encode('ascii', 'ignore').decode('ascii')
    #print("Penghilangan EMOJI :"+text) 
    #text=re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    #text = re.sub(emoji_pattern, '', text, flags=re.MULTILINE)
    #print("Penghilangan EMOJI :"+text)
#==============================MENGHILANGKAN HASHTAG=======================
    #pattern = '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)' #ini kode yang diubah
    #text = re.sub(pattern,'',text, flags=re.MULTILINE)
    #print("Penghilangan Username dan Hashtag :"+text)
#===================================MENGHILANGKAN URL===========================
    pattern = r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*'
    text = re.sub(pattern,'',text, flags=re.MULTILINE)
    print("Penghilangan URL :"+text)
#=======================MENGHILANGKAN TANDA BACA===========================
    remove = string.punctuation
    kd = ' '.join(word.strip(remove) for word in text.split())
    text = kd
    print("Penghilangan Tanda Baca :"+text)
    
#===================================MENGHILANGKAN STOPWORDS===========================
    text = stopword.remove(text)
    print("Stopword :"+text)
#===================================PROSES STEMMING===========================
    text = stemmer.stem(text)
    print("Stemming :"+text)
    
    fw.write(text+"\n")
fw.close()

