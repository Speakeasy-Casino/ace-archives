#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from acquire import *
from prepare_nlp_josh import *
import env
import json
from requests import get
from json.decoder import JSONDecodeError
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
#You can also call the Porter by nltk.porter.PorterStemmer
from nltk.stem.snowball import EnglishStemmer


from nltk.stem import WordNetLemmatizer
from nltk.book import *
from nltk.text import Text


import env


# In[4]:


def get_avg_count_plot():
    links = get_links()

    #Run this to get new data
    df = get_repos(links.href)
    df = clean_languages(df)
    df.dropna(inplace=True)
    df = clean_languages(df)
    df['readme_length'] = df['readme_contents'].apply(len)
    avg_len_ruby =df[df['language']== 'Ruby']['readme_length'].mean()
    avg_len_python= df[df['language']== 'Python']['readme_length'].mean()
    avg_len_java= df[df['language']== 'Java']['readme_length'].mean()
    avg_len_js = df[df['language']== 'JavaScript']['readme_length'].mean()
    avg_len_c= df[df['language']== 'C_based']['readme_length'].mean()
    avg_len_other = df[df['language']== 'Other']['readme_length'].mean()

    avg_df=pd.DataFrame({'Ruby': [avg_len_ruby],
                        'Python': [avg_len_python],
                        'Java':[avg_len_java],
                        'JavaScript': [avg_len_js],
                        'C_based': [avg_len_c],
                        'Other':[avg_len_other]})

    sns.barplot(avg_df, palette='Paired')
    plt.title('Average Word Count of each Readme by Programming Language', fontsize=15)
    plt.xticks(ticks=[0,1,2,3,4,5], labels=['Ruby','Python', 'Java', 'JavaScript', 'C Based', 'Other'])
    plt.xlabel('Programming Languages', fontsize=12)
    plt.ylabel('Average', fontsize=12)


# In[11]:


def get_ruby_count_plot():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    import pandas as pd

    links = get_links()

    #Run this to get new data
    df = get_repos(links.href)
    df = clean_languages(df)
    df.dropna(inplace=True)
    df = clean_languages(df)

    ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt', 'blackjack', 
                            '21']

    # initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) | set(ADDITIONAL_STOPWORDS)

    # load data into a DataFrame


    # define function to lemmatize and remove stopwords from text
    def lemmatize_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        pos_tags = nltk.pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(token, pos=pos_tag[0].lower()) if pos_tag[0].lower() in ['a', 's', 'r', 'v'] else lemmatizer.lemmatize(token) for token, pos_tag in pos_tags]
        lemmatized_text = ' '.join(lemmatized_tokens)
        return lemmatized_text

    # apply lemmatization and stopwords removal function to DataFrame
    df['lemmatized_text'] = df['readme_contents'].apply(lemmatize_text)

    ruby = df[df['language']== 'Ruby'].lemmatized_text
    python = df[df['language']== 'Python'].lemmatized_text
    java = df[df['language']== 'Java'].lemmatized_text
    javascript = df[df['language']== 'JavaScript'].lemmatized_text
    c_based = df[df['language']== 'C_based'].lemmatized_text
    other = df[df['language']== 'Other'].lemmatized_text
    all_words = df.lemmatized_text

    import pandas as pd
    import re
    import unicodedata
    import nltk
    from nltk.stem import WordNetLemmatizer

    ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt']

    def clean(text_list):
        'A simple function to cleanup text data'
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
        cleaned_texts = []
        for text in text_list:
            cleaned_text = (unicodedata.normalize('NFKD', text)
                 .encode('ascii', 'ignore')
                 .decode('utf-8', 'ignore')
                 .lower())
            words = re.sub(r'[^\w\s]', '', cleaned_text).split()
            cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
            cleaned_texts.append(cleaned_words)
        return cleaned_texts

    ruby= clean(ruby)
    python= clean(python)
    java= clean(java)
    javascript= clean(javascript)
    c_based= clean(c_based)
    other= clean(other)
    all_words = clean(all_words)

    def series_words_func(language):
        flat_lst = []

        # Example 2D list of strings


        lst = language

        for sublist in lst:
            for elem in sublist:
                flat_lst.append(elem)

        # Use list comprehension to flatten the list
        flat_lst = [elem for sublist in lst for elem in sublist]

        # Print the flattened list
        language = flat_lst
        return language


    ruby_words= series_words_func(ruby)
    python_words= series_words_func(python)
    java_words= series_words_func(java)
    javascript_words= series_words_func(javascript)
    c_based_words= series_words_func(c_based)
    other_words= series_words_func(other)
    all_words = series_words_func(all_words)


    ruby_freq = pd.Series(ruby_words).value_counts()
    python_freq = pd.Series(python_words).value_counts()
    java_freq = pd.Series(java_words).value_counts()
    javascript_freq = pd.Series(javascript_words).value_counts()
    c_based_freq = pd.Series(c_based_words).value_counts()
    other_freq = pd.Series(other_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()

    word_counts = (pd.concat([ruby_freq, python_freq, java_freq, javascript_freq, c_based_freq, other_freq, all_freq], axis=1, sort=True)
                    .set_axis(['Ruby', 'Python', 'Java', 'Javascript', 'C_based', 'Other', 'All'], axis=1, inplace=False)
                    .fillna(0)
                    .apply(lambda s: s.astype(int)))

    ruby_df=pd.concat([word_counts[word_counts.Ruby == 0],
               word_counts[word_counts.Python == 0],
               word_counts[word_counts.Java == 0], 
               word_counts[word_counts.Javascript == 0], 
               word_counts[word_counts.C_based == 0],
              word_counts[word_counts.Other == 0]]
             ).sort_values(by=['Ruby'], ascending=False).head(25)
    ruby_df['Ruby'].plot(color='red')
    plt.title('Total Unique Count of Words In Ruby')
    plt.xlabel('Ruby')
    plt.ylabel('Total Count')
    plt.show()


# In[12]:


def get_python_count_plot():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    import pandas as pd

    links = get_links()

    #Run this to get new data
    df = get_repos(links.href)
    df = clean_languages(df)
    df.dropna(inplace=True)
    df = clean_languages(df)

    ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt', 'blackjack', 
                            '21']

    # initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) | set(ADDITIONAL_STOPWORDS)

    # load data into a DataFrame


    # define function to lemmatize and remove stopwords from text
    def lemmatize_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        pos_tags = nltk.pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(token, pos=pos_tag[0].lower()) if pos_tag[0].lower() in ['a', 's', 'r', 'v'] else lemmatizer.lemmatize(token) for token, pos_tag in pos_tags]
        lemmatized_text = ' '.join(lemmatized_tokens)
        return lemmatized_text

    # apply lemmatization and stopwords removal function to DataFrame
    df['lemmatized_text'] = df['readme_contents'].apply(lemmatize_text)

    ruby = df[df['language']== 'Ruby'].lemmatized_text
    python = df[df['language']== 'Python'].lemmatized_text
    java = df[df['language']== 'Java'].lemmatized_text
    javascript = df[df['language']== 'JavaScript'].lemmatized_text
    c_based = df[df['language']== 'C_based'].lemmatized_text
    other = df[df['language']== 'Other'].lemmatized_text
    all_words = df.lemmatized_text

    import pandas as pd
    import re
    import unicodedata
    import nltk
    from nltk.stem import WordNetLemmatizer

    ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt']

    def clean(text_list):
        'A simple function to cleanup text data'
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
        cleaned_texts = []
        for text in text_list:
            cleaned_text = (unicodedata.normalize('NFKD', text)
                 .encode('ascii', 'ignore')
                 .decode('utf-8', 'ignore')
                 .lower())
            words = re.sub(r'[^\w\s]', '', cleaned_text).split()
            cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
            cleaned_texts.append(cleaned_words)
        return cleaned_texts

    ruby= clean(ruby)
    python= clean(python)
    java= clean(java)
    javascript= clean(javascript)
    c_based= clean(c_based)
    other= clean(other)
    all_words = clean(all_words)

    def series_words_func(language):
        flat_lst = []

        # Example 2D list of strings


        lst = language

        for sublist in lst:
            for elem in sublist:
                flat_lst.append(elem)

        # Use list comprehension to flatten the list
        flat_lst = [elem for sublist in lst for elem in sublist]

        # Print the flattened list
        language = flat_lst
        return language


    ruby_words= series_words_func(ruby)
    python_words= series_words_func(python)
    java_words= series_words_func(java)
    javascript_words= series_words_func(javascript)
    c_based_words= series_words_func(c_based)
    other_words= series_words_func(other)
    all_words = series_words_func(all_words)


    ruby_freq = pd.Series(ruby_words).value_counts()
    python_freq = pd.Series(python_words).value_counts()
    java_freq = pd.Series(java_words).value_counts()
    javascript_freq = pd.Series(javascript_words).value_counts()
    c_based_freq = pd.Series(c_based_words).value_counts()
    other_freq = pd.Series(other_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()

    word_counts = (pd.concat([ruby_freq, python_freq, java_freq, javascript_freq, c_based_freq, other_freq, all_freq], axis=1, sort=True)
                    .set_axis(['Ruby', 'Python', 'Java', 'Javascript', 'C_based', 'Other', 'All'], axis=1, inplace=False)
                    .fillna(0)
                    .apply(lambda s: s.astype(int)))

    python_df = pd.concat([word_counts[word_counts.Ruby == 0],
               word_counts[word_counts.Python == 0],
               word_counts[word_counts.Java == 0], 
               word_counts[word_counts.Javascript == 0], 
               word_counts[word_counts.C_based == 0],
              word_counts[word_counts.Other == 0]]
             ).sort_values(by=['Python'], ascending=False).head(25)
    python_df['Python'].plot(color='seagreen')
    plt.title('Total Unique Count of Words In Python')
    plt.xlabel('Python')
    plt.ylabel('Total Count')
    plt.show()


# In[14]:


def get_other_count_plot():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    import pandas as pd

    links = get_links()

    #Run this to get new data
    df = get_repos(links.href)
    df = clean_languages(df)
    df.dropna(inplace=True)
    df = clean_languages(df)

    ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt', 'blackjack', 
                            '21']

    # initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) | set(ADDITIONAL_STOPWORDS)

    # load data into a DataFrame


    # define function to lemmatize and remove stopwords from text
    def lemmatize_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        pos_tags = nltk.pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(token, pos=pos_tag[0].lower()) if pos_tag[0].lower() in ['a', 's', 'r', 'v'] else lemmatizer.lemmatize(token) for token, pos_tag in pos_tags]
        lemmatized_text = ' '.join(lemmatized_tokens)
        return lemmatized_text

    # apply lemmatization and stopwords removal function to DataFrame
    df['lemmatized_text'] = df['readme_contents'].apply(lemmatize_text)

    ruby = df[df['language']== 'Ruby'].lemmatized_text
    python = df[df['language']== 'Python'].lemmatized_text
    java = df[df['language']== 'Java'].lemmatized_text
    javascript = df[df['language']== 'JavaScript'].lemmatized_text
    c_based = df[df['language']== 'C_based'].lemmatized_text
    other = df[df['language']== 'Other'].lemmatized_text
    all_words = df.lemmatized_text

    import pandas as pd
    import re
    import unicodedata
    import nltk
    from nltk.stem import WordNetLemmatizer

    ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt']

    def clean(text_list):
        'A simple function to cleanup text data'
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
        cleaned_texts = []
        for text in text_list:
            cleaned_text = (unicodedata.normalize('NFKD', text)
                 .encode('ascii', 'ignore')
                 .decode('utf-8', 'ignore')
                 .lower())
            words = re.sub(r'[^\w\s]', '', cleaned_text).split()
            cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
            cleaned_texts.append(cleaned_words)
        return cleaned_texts

    ruby= clean(ruby)
    python= clean(python)
    java= clean(java)
    javascript= clean(javascript)
    c_based= clean(c_based)
    other= clean(other)
    all_words = clean(all_words)

    def series_words_func(language):
        flat_lst = []

        # Example 2D list of strings


        lst = language

        for sublist in lst:
            for elem in sublist:
                flat_lst.append(elem)

        # Use list comprehension to flatten the list
        flat_lst = [elem for sublist in lst for elem in sublist]

        # Print the flattened list
        language = flat_lst
        return language


    ruby_words= series_words_func(ruby)
    python_words= series_words_func(python)
    java_words= series_words_func(java)
    javascript_words= series_words_func(javascript)
    c_based_words= series_words_func(c_based)
    other_words= series_words_func(other)
    all_words = series_words_func(all_words)


    ruby_freq = pd.Series(ruby_words).value_counts()
    python_freq = pd.Series(python_words).value_counts()
    java_freq = pd.Series(java_words).value_counts()
    javascript_freq = pd.Series(javascript_words).value_counts()
    c_based_freq = pd.Series(c_based_words).value_counts()
    other_freq = pd.Series(other_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()

    word_counts = (pd.concat([ruby_freq, python_freq, java_freq, javascript_freq, c_based_freq, other_freq, all_freq], axis=1, sort=True)
                    .set_axis(['Ruby', 'Python', 'Java', 'Javascript', 'C_based', 'Other', 'All'], axis=1, inplace=False)
                    .fillna(0)
                    .apply(lambda s: s.astype(int)))

    other_df= pd.concat([word_counts[word_counts.Ruby == 0],
               word_counts[word_counts.Python == 0],
               word_counts[word_counts.Java == 0], 
               word_counts[word_counts.Javascript == 0], 
               word_counts[word_counts.C_based == 0],
              word_counts[word_counts.Other == 0]]
             ).sort_values(by=['Other'], ascending=False).head(25)
    other_df['Other'].plot(color='rebeccapurple')
    plt.title('Total Unique Count of Words In Other')
    plt.xlabel('Other')
    plt.ylabel('Total Count')
    plt.show()


# In[15]:


def get_java_count_plot():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    import pandas as pd

    links = get_links()

    #Run this to get new data
    df = get_repos(links.href)
    df = clean_languages(df)
    df.dropna(inplace=True)
    df = clean_languages(df)

    ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt', 'blackjack', 
                            '21']

    # initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) | set(ADDITIONAL_STOPWORDS)

    # load data into a DataFrame


    # define function to lemmatize and remove stopwords from text
    def lemmatize_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        pos_tags = nltk.pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(token, pos=pos_tag[0].lower()) if pos_tag[0].lower() in ['a', 's', 'r', 'v'] else lemmatizer.lemmatize(token) for token, pos_tag in pos_tags]
        lemmatized_text = ' '.join(lemmatized_tokens)
        return lemmatized_text

    # apply lemmatization and stopwords removal function to DataFrame
    df['lemmatized_text'] = df['readme_contents'].apply(lemmatize_text)

    ruby = df[df['language']== 'Ruby'].lemmatized_text
    python = df[df['language']== 'Python'].lemmatized_text
    java = df[df['language']== 'Java'].lemmatized_text
    javascript = df[df['language']== 'JavaScript'].lemmatized_text
    c_based = df[df['language']== 'C_based'].lemmatized_text
    other = df[df['language']== 'Other'].lemmatized_text
    all_words = df.lemmatized_text

    import pandas as pd
    import re
    import unicodedata
    import nltk
    from nltk.stem import WordNetLemmatizer

    ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt']

    def clean(text_list):
        'A simple function to cleanup text data'
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
        cleaned_texts = []
        for text in text_list:
            cleaned_text = (unicodedata.normalize('NFKD', text)
                 .encode('ascii', 'ignore')
                 .decode('utf-8', 'ignore')
                 .lower())
            words = re.sub(r'[^\w\s]', '', cleaned_text).split()
            cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
            cleaned_texts.append(cleaned_words)
        return cleaned_texts

    ruby= clean(ruby)
    python= clean(python)
    java= clean(java)
    javascript= clean(javascript)
    c_based= clean(c_based)
    other= clean(other)
    all_words = clean(all_words)

    def series_words_func(language):
        flat_lst = []

        # Example 2D list of strings


        lst = language

        for sublist in lst:
            for elem in sublist:
                flat_lst.append(elem)

        # Use list comprehension to flatten the list
        flat_lst = [elem for sublist in lst for elem in sublist]

        # Print the flattened list
        language = flat_lst
        return language


    ruby_words= series_words_func(ruby)
    python_words= series_words_func(python)
    java_words= series_words_func(java)
    javascript_words= series_words_func(javascript)
    c_based_words= series_words_func(c_based)
    other_words= series_words_func(other)
    all_words = series_words_func(all_words)


    ruby_freq = pd.Series(ruby_words).value_counts()
    python_freq = pd.Series(python_words).value_counts()
    java_freq = pd.Series(java_words).value_counts()
    javascript_freq = pd.Series(javascript_words).value_counts()
    c_based_freq = pd.Series(c_based_words).value_counts()
    other_freq = pd.Series(other_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()

    word_counts = (pd.concat([ruby_freq, python_freq, java_freq, javascript_freq, c_based_freq, other_freq, all_freq], axis=1, sort=True)
                    .set_axis(['Ruby', 'Python', 'Java', 'Javascript', 'C_based', 'Other', 'All'], axis=1, inplace=False)
                    .fillna(0)
                    .apply(lambda s: s.astype(int)))

    java_df=pd.concat([word_counts[word_counts.Ruby == 0],
               word_counts[word_counts.Java == 0], 
               word_counts[word_counts.Javascript == 0], 
               word_counts[word_counts.C_based == 0],
              word_counts[word_counts.Other == 0]]
             ).sort_values(by=['Java'], ascending=False).head(25)
    java_df['Java'].plot(color='black')
    plt.title('Total Unique Count of Words In Java')
    plt.xlabel('Java')
    plt.ylabel('Total Count')


# In[16]:


def get_c_count_plot():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    import pandas as pd

    links = get_links()

    #Run this to get new data
    df = get_repos(links.href)
    df = clean_languages(df)
    df.dropna(inplace=True)
    df = clean_languages(df)

    ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt', 'blackjack', 
                            '21']

    # initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) | set(ADDITIONAL_STOPWORDS)

    # load data into a DataFrame


    # define function to lemmatize and remove stopwords from text
    def lemmatize_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        pos_tags = nltk.pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(token, pos=pos_tag[0].lower()) if pos_tag[0].lower() in ['a', 's', 'r', 'v'] else lemmatizer.lemmatize(token) for token, pos_tag in pos_tags]
        lemmatized_text = ' '.join(lemmatized_tokens)
        return lemmatized_text

    # apply lemmatization and stopwords removal function to DataFrame
    df['lemmatized_text'] = df['readme_contents'].apply(lemmatize_text)

    ruby = df[df['language']== 'Ruby'].lemmatized_text
    python = df[df['language']== 'Python'].lemmatized_text
    java = df[df['language']== 'Java'].lemmatized_text
    javascript = df[df['language']== 'JavaScript'].lemmatized_text
    c_based = df[df['language']== 'C_based'].lemmatized_text
    other = df[df['language']== 'Other'].lemmatized_text
    all_words = df.lemmatized_text

    import pandas as pd
    import re
    import unicodedata
    import nltk
    from nltk.stem import WordNetLemmatizer

    ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt']

    def clean(text_list):
        'A simple function to cleanup text data'
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
        cleaned_texts = []
        for text in text_list:
            cleaned_text = (unicodedata.normalize('NFKD', text)
                 .encode('ascii', 'ignore')
                 .decode('utf-8', 'ignore')
                 .lower())
            words = re.sub(r'[^\w\s]', '', cleaned_text).split()
            cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
            cleaned_texts.append(cleaned_words)
        return cleaned_texts

    ruby= clean(ruby)
    python= clean(python)
    java= clean(java)
    javascript= clean(javascript)
    c_based= clean(c_based)
    other= clean(other)
    all_words = clean(all_words)

    def series_words_func(language):
        flat_lst = []

        # Example 2D list of strings


        lst = language

        for sublist in lst:
            for elem in sublist:
                flat_lst.append(elem)

        # Use list comprehension to flatten the list
        flat_lst = [elem for sublist in lst for elem in sublist]

        # Print the flattened list
        language = flat_lst
        return language


    ruby_words= series_words_func(ruby)
    python_words= series_words_func(python)
    java_words= series_words_func(java)
    javascript_words= series_words_func(javascript)
    c_based_words= series_words_func(c_based)
    other_words= series_words_func(other)
    all_words = series_words_func(all_words)


    ruby_freq = pd.Series(ruby_words).value_counts()
    python_freq = pd.Series(python_words).value_counts()
    java_freq = pd.Series(java_words).value_counts()
    javascript_freq = pd.Series(javascript_words).value_counts()
    c_based_freq = pd.Series(c_based_words).value_counts()
    other_freq = pd.Series(other_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()

    word_counts = (pd.concat([ruby_freq, python_freq, java_freq, javascript_freq, c_based_freq, other_freq, all_freq], axis=1, sort=True)
                    .set_axis(['Ruby', 'Python', 'Java', 'Javascript', 'C_based', 'Other', 'All'], axis=1, inplace=False)
                    .fillna(0)
                    .apply(lambda s: s.astype(int)))

    c_df= pd.concat([word_counts[word_counts.Ruby == 0],
               word_counts[word_counts.Java == 0], 
               word_counts[word_counts.Javascript == 0], 
               word_counts[word_counts.C_based == 0],
              word_counts[word_counts.Other == 0]]
             ).sort_values(by=['C_based'], ascending=False).head(25)
    c_df['C_based'].plot(color='orange')
    plt.title('Total Unique Count of Words In C Based')
    plt.xlabel('C Based')
    plt.ylabel('Total Count')


# In[17]:


def get_js_count_plot():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    import pandas as pd

    links = get_links()

    #Run this to get new data
    df = get_repos(links.href)
    df = clean_languages(df)
    df.dropna(inplace=True)
    df = clean_languages(df)

    ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt', 'blackjack', 
                            '21']

    # initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) | set(ADDITIONAL_STOPWORDS)

    # load data into a DataFrame


    # define function to lemmatize and remove stopwords from text
    def lemmatize_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        pos_tags = nltk.pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(token, pos=pos_tag[0].lower()) if pos_tag[0].lower() in ['a', 's', 'r', 'v'] else lemmatizer.lemmatize(token) for token, pos_tag in pos_tags]
        lemmatized_text = ' '.join(lemmatized_tokens)
        return lemmatized_text

    # apply lemmatization and stopwords removal function to DataFrame
    df['lemmatized_text'] = df['readme_contents'].apply(lemmatize_text)

    ruby = df[df['language']== 'Ruby'].lemmatized_text
    python = df[df['language']== 'Python'].lemmatized_text
    java = df[df['language']== 'Java'].lemmatized_text
    javascript = df[df['language']== 'JavaScript'].lemmatized_text
    c_based = df[df['language']== 'C_based'].lemmatized_text
    other = df[df['language']== 'Other'].lemmatized_text
    all_words = df.lemmatized_text

    import pandas as pd
    import re
    import unicodedata
    import nltk
    from nltk.stem import WordNetLemmatizer

    ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt']

    def clean(text_list):
        'A simple function to cleanup text data'
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
        cleaned_texts = []
        for text in text_list:
            cleaned_text = (unicodedata.normalize('NFKD', text)
                 .encode('ascii', 'ignore')
                 .decode('utf-8', 'ignore')
                 .lower())
            words = re.sub(r'[^\w\s]', '', cleaned_text).split()
            cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
            cleaned_texts.append(cleaned_words)
        return cleaned_texts

    ruby= clean(ruby)
    python= clean(python)
    java= clean(java)
    javascript= clean(javascript)
    c_based= clean(c_based)
    other= clean(other)
    all_words = clean(all_words)

    def series_words_func(language):
        flat_lst = []

        # Example 2D list of strings


        lst = language

        for sublist in lst:
            for elem in sublist:
                flat_lst.append(elem)

        # Use list comprehension to flatten the list
        flat_lst = [elem for sublist in lst for elem in sublist]

        # Print the flattened list
        language = flat_lst
        return language


    ruby_words= series_words_func(ruby)
    python_words= series_words_func(python)
    java_words= series_words_func(java)
    javascript_words= series_words_func(javascript)
    c_based_words= series_words_func(c_based)
    other_words= series_words_func(other)
    all_words = series_words_func(all_words)


    ruby_freq = pd.Series(ruby_words).value_counts()
    python_freq = pd.Series(python_words).value_counts()
    java_freq = pd.Series(java_words).value_counts()
    javascript_freq = pd.Series(javascript_words).value_counts()
    c_based_freq = pd.Series(c_based_words).value_counts()
    other_freq = pd.Series(other_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()

    word_counts = (pd.concat([ruby_freq, python_freq, java_freq, javascript_freq, c_based_freq, other_freq, all_freq], axis=1, sort=True)
                    .set_axis(['Ruby', 'Python', 'Java', 'Javascript', 'C_based', 'Other', 'All'], axis=1, inplace=False)
                    .fillna(0)
                    .apply(lambda s: s.astype(int)))

    js_df= pd.concat([word_counts[word_counts.Ruby == 0],
               word_counts[word_counts.Java == 0], 
               word_counts[word_counts.Javascript == 0], 
               word_counts[word_counts.C_based == 0],
              word_counts[word_counts.Other == 0]]
             ).sort_values(by=['Javascript'], ascending=False).head(25)
    js_df['Javascript'].plot(color='magenta')
    plt.title('Total Unique Count of Words In JavaScript')
    plt.xlabel('JavaScript')
    plt.ylabel('Total Count')


# In[ ]:




