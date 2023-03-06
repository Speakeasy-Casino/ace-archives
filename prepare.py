import os
import json
from typing import Dict, List, Optional, Union, cast
import requests
import acquire as a
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
import requests
import unicodedata
import nltk
from textblob import TextBlob, Word
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

def get_links(csv_file):
    df = pd.read_csv(csv_file)
    return df

def get_languages(df):
    ndf = a.get_dataframe(df)
    return ndf
    
def clean_languages(df):
    df.language[df.language == 'Jupyter Notebook'] = 'Python'
    df.language[(df.language == 'Objective-C') 
             |(df.language == 'C#')
            |(df.language == 'C++') 
            |(df.language == 'C')] = 'C_based'
    df.language[(df.language == 'Swift') 
            | (df.language == 'CSS') 
            | (df.language == 'HTML')
            | (df.language == 'TypeScript')
            | (df.language == 'Lua')
            | (df.language == 'Kotlin')
            | (df.language == 'Vue')
            | (df.language == 'Rust')
            | (df.language == 'PHP')
            | (df.language == 'Go')
            | (df.language == 'Dart')
            | (df.language == 'Roff')
            | (df.language == 'Haskell')
            | (df.language == 'Visual Basic')
            | (df.language == 'Solidity')
            | (df.language == 'Clojure')
            | (df.language == 'Elixir')
            | (df.language == 'Shell')] = 'Other'
    return df

def drop_blanks(df):
    df = df[df.readme_contents != '']
    return df

def split(input_string):
    '''
    Creates and uses a snowball stemmer for readme data
    '''
    splitter = nltk.stem.SnowballStemmer('english')
    
    stem = [splitter.stem(word) for word in input_string.split()]
    split_string = ' '.join(stem)
    
    return split_string

def lemmatized(input_string):
    """
    Takes an input string and lemmatizes it.
    Please do not stem and lemmatize the same string.
    """
    #Creates the lemmatizer object
    wnl = nltk.stem.WordNetLemmatizer()
    
    #Makes lemmatade
    lemmas = [wnl.lemmatize(word) for word in input_string.split()]
    lemmatized_string = ' '.join(lemmas)
    
    return lemmatized_string

def remove_stopwords(input_string, extra_words=None, exclude_words=None):
    """
    This function takes an input and removes stopwords. You can add or remove words witht the extra_words or exclude_words args.
    """
    
    stopword_list = stopwords.words('english')
    if extra_words != None:
        stopword_list.extend(extra_words)
    if exclude_words != None:
        if type(extra_words) == str:
            stopword_list.remove(exclude_words)
        if type(extra_words) == list:
            for word in exclude_words:
                stopword_list.remove(word)
    words = input_string.split()
    filtered_words = [w for w in words if w not in stopword_list]
    string_without_stopwords = ' '.join(filtered_words)

    return string_without_stopwords


def basic_clean(input_string):
    """
    This function takes in a string and applies basic cleaning to it.
    """
    #Changes all characters to their lower case.
    input_string = input_string.lower()
    
    input_string = unicodedata.normalize('NFKD', input_string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    #Removes special characters
    input_string = re.sub(r"[^a-z0-9\s]", '', input_string)
    input_string = re.sub(r'\n', ' ', input_string)
    input_string = re.sub(r'\s{2,}', ' ', input_string)
    return input_string

def tokenized(input_string):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    return tokenizer.tokenize(input_string, return_str=True)