import os
import json
from typing import Dict, List, Optional, Union, cast
import requests
import acquire as a
import re
import numpy as np
import pandas as pd

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