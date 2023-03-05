import re
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.snowball import EnglishStemmer

import pandas as pd
import numpy as np
import requests
import os
import json
import unicodedata
import nltk

def tokenized(input_string, tokenize_tool=1):
    """
    Input:
    This function takes in a string and tokenizer tool argument and returns a list of tokens.
    tokenize_tool=1: ToktokTokenizer
    tokenize_tool=2: word_tokenizer
    tokenize_tool=3: sent_tokenizer
    """
    
    if tokenize_tool==1:
        tokenizer = ToktokTokenizer()
        return tokenizer.tokenize(input_string)
    elif tokenize_tool == 2:
        return word_tokenize(input_string)
    elif tokenize_tool == 3:
        return sent_tokenize(input_string)

def basic_clean(input_string):
    """
    This function takes in a string and applies basic cleaning to it.
    """
    #Changes all characters to their lower case.
    input_string = input_string.lower()

    input_string = unicodedata.normalize('NFKD', input_string).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    #Removes special characters
    input_string = re.sub(r"[^a-z0-9\s]", '', input_string)

    return input_string

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

def stemmertize_tool(input_list, stemmer_type=1):
    """
    Input a list of words to stemmertize. Returns a list.
    stemmer_type=1 - PorterStemmer
    stemmer_type=2 - EnglishStemmer
    stemmer_type=3 - SnowballStemmer("english")
    """
    
    if stemmer_type ==1:
        stemmer = PorterStemmer()
    elif stemmer_type ==2:
        stemmer = EnglishStemmer()
    elif stemmer_type ==3:
        stemmer = SnowballStemmer("english")
    
    return [stemmer.stem(word) for word in input_list]

def extract_proper_nouns(quote):
    """
    Intakes a string and returns words tagged as proper nouns.
    """
    words = word_tokenize(quote)
    tags = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tags, binary=True)
    return set(" ".join(i[0] for i in t)for t in tree if hasattr(t, "label") and t.label() == "NE")

def ngrams_creator(input_list, n_grams = 2):
    """
    This function takes in a list and returns a list of grams.
    """
    ngrams = nltk.ngrams(input_list, n_grams)
    return list(ngrams)

