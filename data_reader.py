#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import json

import html
import re
import string
import preprocessor as p

p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY)


# In[2]:


DATA_1_FILEPATH = ["//data/data1/training_set.json", "/data/data1/test_set.json"]
DATA_2_FILEPATH = "/data/data2/tweets_labelled_09042020_16072020.csv"
DATA_3_FILEPATH = "/data/data3/data3_final.csv"
DATA_4_FILEPATH = ["/data/data4/stocktwits_data_ALL.csv", "/data/data4/stocktwits_data_ALL_cleaned.csv"]


# In[3]:


def preprocess_data(stocktwit_df, col_name):
    def remove_emoji(tweets):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', tweets)


    stocktwit_df[col_name] = stocktwit_df[col_name].astype(str)

    # Removing all tickers from comments
    stocktwit_df[col_name] = stocktwit_df[col_name].str.replace(r'([$][a-zA-z\.]{1,8})', '')

    # Make all sentences small letters
    stocktwit_df[col_name] = stocktwit_df[col_name].str.lower()

    # Converting HTML to UTF-8
    stocktwit_df[col_name] = stocktwit_df[col_name].apply(html.unescape)

    # Removing hastags, mentions, pagebreaks, handles
    # Keeping the words behind hashtags as they may provide useful information about the comments e.g. #Bullish #Lambo
    stocktwit_df[col_name] = stocktwit_df[col_name].str.replace(r'(@[^\s]+|[#]|[$])', ' ')  # Replace '@', '$' and '#...'
    stocktwit_df[col_name] = stocktwit_df[col_name].str.replace(r'(\n|\r)', ' ')  # Replace page breaks

    # Removing https, www., any links etc
    stocktwit_df[col_name] = stocktwit_df[col_name].str.replace(r'((https:|http:)[^\s]+|(www\.)[^\s]+)', ' ')

    # Removing all numbers
    stocktwit_df[col_name] = stocktwit_df[col_name].str.replace(r'[\d]', '')

    # Remove emoji
    stocktwit_df[col_name] = stocktwit_df[col_name].apply(lambda row: remove_emoji(row))

    # Remove punctuations
    stocktwit_df[col_name] = stocktwit_df[col_name].str.translate(str.maketrans('', '', string.punctuation+"“”‘’…•—"))

    # All additional cleaning
    stocktwit_df[col_name] = stocktwit_df[col_name].apply(lambda row: p.clean(row))

    # Remove whitespaces
    stocktwit_df[col_name] = stocktwit_df[col_name].apply(lambda row: " ".join(row.split()))

    # Remove empty rows
    stocktwit_df = stocktwit_df[stocktwit_df[col_name].str.contains(r'^\s*$') == False]

    return stocktwit_df


# In[4]:


stop_word= ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '@', '#']

def to_tokens(data):
    sentence_token = [s.split(' ') for s in data] 
    return sentence_token


# In[26]:


def read_data1(mode, to_clean= True):

    cwd = os.getcwd()
    train_path = cwd + DATA_1_FILEPATH[0]
    test_path = cwd + DATA_1_FILEPATH[1]

    with open(train_path, "r") as f:
        data = f.read()
        data1 = json.loads(data)

    with open(test_path, "r") as f:
        data = f.read()
        test_data = json.loads(data)

    data1.extend(test_data)
    data1 = pd.DataFrame(data1)
    
    if(to_clean):
        data1 = preprocess_data(pd.DataFrame(data1), "tweet")
        
    if(mode=="dataframe"):
        data1_X = [item.lower() for item in data1['tweet']]
        data1_y_class = [1 if float(item)>0 else 0 for item in data1["sentiment"]]
        
        data1 = pd.DataFrame()
        data1["text_cleaned"] = data1_X
        data1["Label"] = data1_y_class
        return data1
    
    elif(mode=="list"):
        data1_X = [item.lower() for item in data1['tweet']]
        data1_X = to_tokens(data1_X)
        data1_y_class = [1 if float(item)>0 else 0 for item in data1["sentiment"]]

        return data1_X, data1_y_class


# In[6]:


def read_data2(mode, to_clean= True):
    
    cwd = os.getcwd()
    path = cwd + DATA_2_FILEPATH
    
    data2 = pd.read_csv(path, sep=";")
    data2 = data2[data2['sentiment'].notna()]
    data2 = data2[data2['sentiment']!='neutral']
    
    if(to_clean):
        data2 = preprocess_data(data2, "text")
    
    if(mode=="dataframe"):
        data2_X = [item.lower() for item in data2['text']]
        data2_y_class = [1 if item=="positive" else 0 for item in data2["sentiment"]]
        
        data2 = pd.DataFrame()
        data2["text_cleaned"] = data2_X
        data2["Label"] = data2_y_class
        return data2
    
    elif(mode=="list"):
        data2_X = [item.lower() for item in data2['text']]
        data2_X = to_tokens(data2_X)
        data2_y_class = [1 if item=="positive" else 0 for item in data2["sentiment"]]
        
        return data2_X, data2_y_class


# In[7]:


def read_data3(mode, to_clean= True):
    
    cwd = os.getcwd()
    path = cwd + DATA_3_FILEPATH
    
    data3 = pd.read_csv(path)
    
    if(to_clean):
        data3 = preprocess_data(data3, "Message")
        
    if(mode=="dataframe"):
        data3_X = [item.lower() for item in data3['Message']]
        data3_y_class = [int(sentiment) for sentiment in data3['Annotator1']]
        
        data3 = pd.DataFrame()
        data3["text_cleaned"] = data3_X
        data3["Label"] = data3_y_class
        return data3
    
    elif(mode=="list"):
        data3_X = [item.lower() for item in data3['Message']]
        data3_X = to_tokens(data3_X)
        data3_y_class = [int(sentiment) for sentiment in data3['Annotator1']]
        
        return data3_X, data3_y_class


# In[8]:


def read_data4(mode, to_clean= True, sample_size=10000, seed=100):
    
    cwd = os.getcwd()
    
    if(to_clean):
        path = cwd + DATA_4_FILEPATH[1]
    else:
        path = cwd + DATA_4_FILEPATH[0]
        
    data4 = pd.read_csv(path)
    
    if(mode=="lexicon"):
        return data4
    
    elif(mode=="bert"):
        data4['Label'] = [1 if sentiment=="Bullish" else 0 for sentiment in data4['Sentiment']]
        data4['text_cleaned'] = data4['Message']
        data4 = data4[["text_cleaned", "Label"]]
        
        data4 = data4.sample(n=sample_size)
        train_df = data4.sample(frac=0.8,random_state=seed)
        val_df = data4.drop(train_df.index)
        return train_df, val_df

