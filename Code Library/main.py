# Python Script to Build Web Application

# Import Libraries and Dataset
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split


# Neural Network Models
#import tensorflow as tf
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Embedding, LSTM, Dense
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Bidirectional, Dropout
#from keras.callbacks import EarlyStopping
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy

# Libraries For Data Preparation
import os
import re
import sys
import pandas as pd
import numpy as np
import nltk
import contractions
import string
from collections import defaultdict, Counter
from string import punctuation
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer, word_tokenize, TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tabulate import tabulate
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
import xgboost as xgb


# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')

# NLTK Downloads
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')


# Create Functions For Cleaning Tweets

sw = stopwords.words("english")
lemmatizer = WordNetLemmatizer()  
# Create Function for Cleaning Tweet

def data_preparation_pipeline(tweet):
    tweet = str(tweet).lower() # Convert tweet to lowercase
    tweet = re.sub('\[.*?\]', '', tweet) # Remove text within square brackets
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet) # Remove URLs starting with "http://" or "https://"
    tweet = re.sub('https?://\S+|www\.\S+', '', tweet) # Remove URLs starting with "https://" or "www."
    tweet = tweet.replace('&gt;', '>') # Replaces html sign for greater than sign
    tweet = tweet.replace('&le;', '<') # Replaces html sign for less than sign
    tweet = tweet.replace('&equals;', 'equals') # Replaces html sign for equals
    tweet = tweet.replace('&amp;', 'and') # Replaces html sign for equals
    tweet = tweet.replace('&#8220;', '"') # Replaces html sign for quotation mark
    tweet = re.sub('<.*?>+', '', tweet) # Remove HTML tags
    punctuation_to_keep = ['#', '@'] # Keep Hashtag and @ symbols
    tweet = ''.join(char for char in tweet if char.isalnum() or char in punctuation_to_keep or char.isspace()) # Remove punctuation marks
    tweet = re.sub('\n', '', tweet) # Remove newlines
    tweet = re.sub(r'#\d+', '', tweet) # Remove hashtags followed by digits
    tweet = re.sub(r'[0-9]', '', tweet) # Remove digits
    tweet = re.sub(r'#', '', tweet) # Remove hashtags
    tweet = re.sub('\w*\d\w*', '', tweet) # Remove words containing numbers
    tweet = tweet.replace('rt', '')  # Remove "rt" from the tweet
    tweet = tweet.replace('RT', '')  # Remove "RT" from the tweet
    tweet = re.sub('VIDEO:','', tweet) # Remove Video label from the tweet
    tweet = re.sub(r"\s+",' ', tweet) # Remove extra spaces
    tweet = contractions.fix(tweet) # Replace contractions with their expanded forms
    return tweet

# Function to filter out tweets with "#" in front of "@" symbols
def ignore_special_hashtags(tweet):
    return not ('#@' in tweet)

def ignore_special_punc_func(text):
    text1 = [ignore_special_hashtags(word) for word in text]
    return text1

def lemmatize(word):
    tweet = word.split()
    replaced_words = []
    lemmatizer = WordNetLemmatizer()

    for token, tag in pos_tag(tweet):
        pos = tag[0].lower()
        lm = lemmatizer.lemmatize(token, pos)
        replaced_words.append(lm)
        tweet = " ".join(replaced_words)
        print(tweet)
    return tweet

# Remove stop words
def remove_stop(tokens) :
    tokens = [file for file in tokens if file not in sw]
    return(tokens)

def spellcheck(text):
    #text = word.split()
    replaced_words = []
    spell = SpellChecker()
    #spell.word_frequency.load_text_file('load file')
    #spell.word_frequency.load_words(['''add any words that might be corrected but it shouldn't be'''])
    for token in text:
        correct_word = spell.correction(token)
        replaced_words.append(token)
        print(correct_word, replaced_word)
        corpus = " ".join(replaced_words)
    return corpus

def reduce_lengthening(tweet):
    ''' 
    Reduces repeated characters that occurs more than twice in a word.
    Example:
        meeet > meet
    '''
    pattern = re.compile(r"(.)\1(2,)")
    orig_map = {}
    for word in tweet:
        reduced_word = pattern.sub(r"\1\1", word)
        orig_map[word] - reduced_word
    return orig_map

# Tokenize the given text
def tokenize(text) :  
    text = [file.lower().strip() for file in text.split()]
    return(text)

# Function to tokenize tweets using TweetTokenizer
def tweet_tokenize(text):
    tknzr = TweetTokenizer(reduce_len=True)
    return tknzr.tokenize(text)

def data_preparation_pipeline_func(text):
    text1 = [data_preparation_pipeline(word) for word in text]
    return text1

# Lemmatization Function - Source: https://blog.devgenius.io/preprocessing-twitter-dataset-using-nltk-approach-1beb9a338cc1

# convert stringified list to list
def strg_list_to_list(strg_list):
  return strg_list.strip("[]").replace("'","").replace('"',"").replace(",","").split() 


def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

#lemmatize requires list input
def lemmatize(unkn_input):
    if (isinstance(unkn_input,list)):
      list_input=unkn_input
    if (isinstance(unkn_input,str)):
      list_input=strg_list_to_list(unkn_input)
    list_sentence = [item.lower() for item in list_input]
    nltk_tagged = nltk.pos_tag(list_sentence)  
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])),nltk_tagged)
    
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:        
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        #" ".join(lemmatized_sentence)
    return lemmatized_sentence

def preprocess(text):
    corpus = []
    for token in text:
        if token not in sw:
            corpus.append(token)
           
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(' '.join(corpus))
 
    text = ' '.join(tokens)
    num_pattern = re.compile('[0-9]+')
    num_pattern.sub(r'', text)
   
    return num_pattern.sub(r'', text)

def preprocess1(text):
    corpus = []
    for token in text:
        if token not in sw:
            corpus.append(token)
           
    #tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S')
    tokens = ' '.join(corpus)
 
    text = ''.join(tokens)
    return text

# Compile text into a pipeline
def prepare(text, pipeline) : 
    tokens = str(text)
    for transform in pipeline : 
        tokens = transform(tokens)
    return(tokens)

##################

st.header("Hate Speech and Offensive Language Detection through Sentiment Analysis App")
#st.text_input("Enter your Name: ", key="name")
df1 = pd.read_csv("https://github.com/katie-hu/ADS599-Capstone/blob/main/Data/Prepared_Data.csv")


# Load Functions and Saved Best Model 

# Set the Label to be Numerical - The index of 'negative' is 0, 'neutral' is 1, and 'positive' is 2 in this list
label = df1['sentiment']
sentiment_ordering = ['negative', 'neutral', 'positive']
y = label.apply(lambda x: sentiment_ordering.index(x))
     

# Input column
X = df1['clean_tweet']

# Splitting of Data

X_train, X_test, y_train, y_test = train_test_split(df1['clean_tweet'], y, test_size = .15, stratify = y, random_state = 1025)

# Use TfidfVectorizer to convert text to a matrix of TF-IDF features
vec = TfidfVectorizer(strip_accents = 'unicode', analyzer = 'word', ngram_range = (1,3))
vec.fit(X_train)
X_train = vec.transform(X_train)
X_test = vec.transform(X_test)

xgb_model=xgb.XGBClassifier(
    learning_rate=0.1,
    max_depth=7,
    n_estimators=80,
    use_label_encoder=False,
    eval_metric='auc' )

xgb_model_vectorizer = xgb_model.fit(X_train, y_train)
# Tokenize preprocessing for LSTM model
#tokenizer = Tokenizer()
#tokenizer.fit_on_texts(X)
#X_n = tokenizer.texts_to_sequences(X)

# Pad the sequences for same length
#maxlen = 120
#X_n = pad_sequences(X_n, maxlen=maxlen)

# num of classes
#num_classes = len(sentiment_ordering)

#X_train_n, X_test_n, y_train, y_test = train_test_split(X_n, y, test_size = .15, stratify = y, random_state = 1025)
# Add Early Stopping
#early_stop = EarlyStopping(monitor="val_loss",patience=5,verbose=True)

# Load Model
#model = Sequential()
#model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=maxlen))
#model.add(LSTM(100))
#model.add(Dense(128, activation='relu'))
#model.add(Dense(units=num_classes, activation='sigmoid'))

# Compile the model
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.load_model("lstm (2).json")

# Show dataframe on Web


# Set Up Input Tweet
user_input = st.text_area("Enter Tweet Data")


# Run data preparation pipeline
my_pipeline = [tokenize, remove_stop, data_preparation_pipeline_func, lemmatize]
test_series = pd.Series([user_input])
test_cleaned = test_series.apply(prepare, pipeline = my_pipeline)

# Apply filter to clean tokens for special punctuation
test_cleaned = test_cleaned[test_cleaned.apply(ignore_special_hashtags)]

# Create a cleaned column of tweets
test_cleaned = test_cleaned.apply(preprocess1)

# Make Prediction Function

if st.button('Predict Sentiment'):
    test_input = vec.transform(test_cleaned)

    res = xgb_model_vectorizer.predict(test_input)[0]
    if res==2:
        st.write(f"Positive Sentiment - No Hate Speech and Offensive Language Detected")
    elif res==1:
        st.write(f"Neutral Sentiment - Hate Speech and Offensive Language Not Detected")
    elif res==0:
        st.write(f"Negative Sentiment - Hate Speech and Offensive Language Detected")


