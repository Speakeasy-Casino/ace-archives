# Speakeasy-Casino's Natural Language Processing Analysis
### By Data Scientist : Andrew Konstans, Joshua Holt, Jacob Panyathong, Brandon Navarrete

<a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-013243.svg?logo=python&logoColor=blue"></a>
<a href="#"><img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=red"></a>
<a href="#"><img alt="NumPy" src="https://img.shields.io/badge/Numpy-2a4d69.svg?logo=numpy&logoColor=black"></a>
<a href="#"><img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-8DF9C1.svg?logo=matplotlib&logoColor=blue"></a>
<a href="#"><img alt="seaborn" src="https://img.shields.io/badge/seaborn-65A9A8.svg?logo=pandas&logoColor=red"></a>
<a href="#"><img alt="sklearn" src="https://img.shields.io/badge/sklearn-4b86b4.svg?logo=scikitlearn&logoColor=black"></a>
<a href="#"><img alt="SciPy" src="https://img.shields.io/badge/SciPy-1560bd.svg?logo=scipy&logoColor=blue"></a>

# :black_joker: Goal

Our team has set out to learn the NLP pipeline while keeping our interest in mind. So we seek to create a M.l model to classify repositories based on their primary coding language only looking at the words in their ReadMe's. We will work on repositories that are focused on the topic "Blackjack"

* We aim to develope a model that can classify github repository programing languages.

* Create a report which is easy to read and interpret, and reproducable.

## :spades: Data Overview

* Our data was webscraped for github, where we collected about 380 urls which lead to individual repositories on github

* These Repo's had specific attributes we tried to follow:
    * In English
    * Most Forked Category
    * Had a Specific Key Word: Blackjack
    
    
# :diamonds: Initial Questions
* What are the Most Common Words in the ReadMe's?

* Does the Length of ReadMe's vary among Programming Languages?

* Do Specific Programming Language Repositories have different amount of Unique Words?

* Are There Some Words that Only show up in Key Programming Languages?

### :clubs: Project Plan / Process
#### :one:   Data Acquisition

<details>
<summary> Gather data from Github </summary>

- Scrape URL's meeting requirements

- Save to local machine

</details>

<details>
<summary> acquire.py </summary>

- Create acquire.py and user-defined functions to import data from local saves. Create dataframes.
</details>


#### :two:   Data Preparation

<details>
<summary> Data Cleaning</summary>

- **Missing values:**
    - Drop NULLS

- **Dropped**
     - We created a feature called `Others` which was a collection of less frequent programming languages. We dropped this.(Hurting the Model)
     
- **NLP**
     - Cleaned, Tokenized, Stemmed, Lemmed to get ready for exploring and modeling
    - select stopwords 
   ( blackjack, java, cards, split, ace
variables, conditional , statements, loops, functions, object, oriented programming, syntax, comments, libraries, frameworks)

- **train,validating,test:**
    - stratify against `language` columns
n-alphanumeric characters.
</details>

        
#### :three:   Exploratory Analysis

<details>
<summary> Questions </summary>

* What are the Most Common Words in the ReadMe's?

* Does the Length of ReadMe's vary among Programming Languages?

* Do Specific Programming Language Repositories have different amount of Unique Words?

* Are There Some Words that Only show up in Key Programming Languages?
</details>
   
 
#### :four:   Modeling Evaluation

<details>
<summary> Models </summary>

* Create Baseline

* Test decision tree, random forest, logistic regression ( select best with given parameter, stem v.s lemm, TF-IDF or Count Vectorizer
</details>


### :medal_sports: Key Findings 
<details>
   
   
<summary> Key Points </summary>
   
- Ruby was the most used and most common language among our programming languages.
- Our best model, given parameters and stopwords, was a decsion tree at 77% accuracy
- We had to drop `Others` column. Our model was not able to capture this very well
- We were able to out perform the baseline.

</details>


# Recommendation
* This model is beating the baseline of just guessing. It is fit for production UNTIL we develope a model that can outperform this one.


:electron: # Next Steps
*  This model took about 380 repo's, more data wouldn't hurt.
* Our "Other" category was a catch-all for the least popular languages. Our model was not successful in this area, why?
* We should explore n-grams, as this may benefit the model


To Reproduce:
1. Acquire a personal access token from https://github.com/settings/tokens
Note: You do not need select any scopes, i.e. leave all the checkboxes unchecked.
2. Save it in your env.py file under the variable "github_token"
3. Save your github username to your env.py file under the variable "github_username"
4. Clone the final_notebook.ipynb, acquire.py, prepare.py, explore.py and modeling.py files to your computer.
5. Ensure you have adequate time and an uninterrupted internet connection for approximately 30 minutes (during the first run.)
6. Run the final_notebook.ipynb.
Note: NLTK may require subsequent downloads listed in error messages.

Acquire list of urls:
We acquired a list of GitHub repositories using BeautifulSoup web scraping package. We entered the word "blackjack" in the search function on the GitHub website and sorted by "Most forks". 

Acquire ReadMe of repositories:
We passed the list of repositories to a function that acquired the primary programming language and text from the ReadMe.

Prepare the data:
During the preparation process we dropped entries with null values. Then we combined all the C programming languages. We kept the top five languages and renamed all the other languages to "other." We cleaned the text by removing non-alphanumeric characters.

