ReadMe - Natural Language Processing:

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