#!/usr/bin/env python
# coding: utf-8

#-------------------------------------------------------------------------
#This py file is being used to perform data preprocessing on dataset to performsentiment analysis
#on movie reviews from Rotten Tomatoes website. The dataset contains two tables named 'rotten_tomatoes_critic_reviews'
#and 'rotten_tomatoes_movies'. First, we merge these two tables into one and dropping any null values and duplicates.
#Then, create user defined function to expand contractions and lemmatize the string to its base form. Feature engineering
#is being used to convert different review score scale into a single scale. Finally, stop words are removed by using nltk
#package and 'movie.csv'exported as csv to be used for data analysis and ML model.

__author__ = "Bryan Vega, Terry Hill, Tony Liao, Srividya Panchagnula"
__credits__ = ["Bryan Vega", "Terry Hill", "Tony Liao", "Srividya Panchagnula"]
__version__ = "0.1"
__maintainer__ = "Bryan Vega, Terry Hill, Tony Liao, Srividya Panchagnula"
__email__ = "bvega4@gmu.edu, thill22@gmu.edu, zliao5@gmu.edu, spanchag@gmu.edu"
__status__ = "Prototype"

#-------------------------------------------------------------------------



import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import pandas as pd
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


#read movie review and movie info data
movie_review = pd.read_csv('rotten_tomatoes_critic_reviews.csv')
movie_info = pd.read_csv('rotten_tomatoes_movies.csv')

#merge info and review data into a consolidated dataframe
movie = pd.merge(movie_info, movie_review)
movie.describe()
movie.isnull().sum()

#remove missing values
movie = movie.dropna()
movie.isnull().sum()

#remove duplicates
movie.drop_duplicates()

# Function to expand contractions
def expand_contractions(text):
    return contractions.fix(text)

# Apply function to the 'review_content' column
movie['review_content'] = movie['review_content'].apply(expand_contractions)

# create an instance of WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# define a function to apply lemmatization to a string
def lemmatize_string(text):
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# apply lemmatization to 'review_content'
movie['review_content'] = movie['review_content'].apply(lemmatize_string)

#remove punctuations
movie['review_content'] = movie['review_content'].str.replace(r'[^\w\s]', '')

#lower case conversion
movie['review_content'] = movie['review_content'].str.lower()

#Review score conversion
def convert_score(score):
    match = re.search('(\d+)(/4|/5|/10)', score)
    if match:
        numerator = int(match.group(1))
        denominator = int(match.group(2)[1:])
        return str(round(numerator / denominator * 10, 1)) #+ '/10'
    else:
        return None
    
movie['review_score'] = movie['review_score'].apply(convert_score)

#drop missing review score
movie = movie.dropna()

# Load the stopwords
stop_words = set(stopwords.words('english'))

# Define a function to remove stopwords
def remove_stopwords(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove the stopwords
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    # Join the filtered tokens back into a string
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# Apply the function to 'review_content'
movie['review_content'] = movie['review_content'].apply(remove_stopwords)

movie.to_csv('movie.csv', index=False)
