import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk

def ngrams_creator(input_string, n_grams = 2):
    """
    This function takes in a list and returns a list of grams.
    """
    ngrams = nltk.ngrams(input_string.split(), n_grams)
    return list(ngrams)

def explore_visual_1(df):
    """
    Visual to show the average word count by programming language
    """
    
    df['readme_length'] = df['readme_contents'].apply(len)
    
    avg_len_ruby =df[df['language']== 'Ruby']['readme_length'].mean()
    avg_len_python= df[df['language']== 'Python']['readme_length'].mean()
    avg_len_java= df[df['language']== 'Java']['readme_length'].mean()
    avg_len_js = df[df['language']== 'JavaScript']['readme_length'].mean()
    avg_len_c= df[df['language']== 'C_based']['readme_length'].mean()
    
    avg_df=pd.DataFrame({'Ruby': [avg_len_ruby],
                    'Python': [avg_len_python],
                    'Java':[avg_len_java],
                    'JavaScript': [avg_len_js],
                    'C_based': [avg_len_c]})
    
    sns.barplot(avg_df, palette='Paired')
    plt.title('Average Word Count of Readme by Programming Language', fontsize=15)
    plt.xticks(ticks=[0,1,2,3,4], labels=['Ruby','Python', 'Java', 'JavaScript', 'C Based'])
    plt.xlabel('Programming Languages', fontsize=12)
    plt.ylabel('Average', fontsize=12)
    
def explore_visual_2(train):
    """
    Creates a visual to show bi grams.
    """
    big_rams_stem = []
    for row in train['readme_stem_no_swords'].apply(ngrams_creator):
        big_rams_stem.extend(row)
        
    bi_stem_series = pd.Series(big_rams_stem)
    
    top_25_readme_bigrams = bi_stem_series.value_counts().head(25)
    top_25_readme_bigrams.sort_values(ascending=True).plot.barh(color='royalblue', width=.9, figsize=(10, 6))

    plt.title('25 Most frequently occuring readme bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_25_readme_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)