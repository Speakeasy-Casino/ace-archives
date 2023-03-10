"""
A module for obtaining repo readme and language data from the github API.

Before using this module, read through it, and follow the instructions marked
TODO.

After doing so, run it like this:

    python acquire.py

To create the `data.json` file that contains the data.
"""
import os
from os import path
import json
from typing import Dict, List, Optional, Union, cast
import requests
import pandas as pd
import numpy as np
from requests import get

from json.decoder import JSONDecodeError
from bs4 import BeautifulSoup

# Visual Imports
import time
from tqdm import tqdm

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token: https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.


headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        if "language" not in repo_info:
            raise Exception(
                "'language' key not round in response\n{}".format(json.dumps(repo_info))
            )
        return repo_info["language"]
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = ""
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data(REPOS) -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


#if __name__ == "__main__":
#    data = scrape_github_data()
#    json.dump(data, open("data.json", "w"), indent=1)

def get_repos(REPOS):
    """
    This function takes a list of github repos, cache the data and returns the url, programming language and readme.
    """
    file = 'data.json'
    
    if os.path.exists(file):
        
        return pd.read_json('data.json')
    else:
        data = scrape_github_data(REPOS)
        json.dump(data, open("data.json", "w"), indent=1)
        df = pd.read_json('data.json')
        
        #Save the data as csv
        df.to_csv("origional_data.csv", index=False)
        return df

def get_links():
    
    """
    Scrapes GitHub for repositories related to the keyword "blackjack" and returns
    a DataFrame containing the href values of the repositories found. If a file "links.csv"
    exists in the current directory, it reads the file and returns the DataFrame instead of
    scraping GitHub. If the file does not exist, it scrapes GitHub and saves the DataFrame
    to the file.
    Returns:
        df (pandas.DataFrame): A DataFrame containing the href values of the repositories
            related to the keyword "blackjack".
    """
    
    # create an empty dataframe
    df = pd.DataFrame()

    # create a list to store hrefs
    hrefs = []
    
    # check if file "links.csv" exists
    if path.exists("links.csv"):
        df = pd.read_csv("links.csv")
        return df
    else:
        
        for i in range(1,40):
            url = f'https://github.com/search?o=desc&p={i}&q=blackjack&s=forks&type=Repositories'
        
            # Create a response based on my headers
            print(url)
            response = get(url, headers=headers)
        
            print(response.ok, response.status_code)
        
            # create soup object
            soup = BeautifulSoup(response.content, 'html.parser')
        
            # create a find_all list
            anchors = soup.find_all('a', class_='v-align-middle')
        
            # list comprehension to get href values for all anchor tags
            for anchor in anchors:
                href = anchor.get('href')
                print(href)
                hrefs.append(href)
            # wait a bit to avoid overloading the API
            time.sleep(20)
        
        # append the hrefs to the data frame
        df = df.append(pd.DataFrame({'href': hrefs}))

        df = df[df.href != '/KillovSky/Iris']
        # save the DataFrame to a CSV file
        df.to_csv("links.csv", index=False)
   
    return df