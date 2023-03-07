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

import env

