#!/usr/bin/env python
# coding: utf-8

#-------------------------------------------------------------------------
#Once the data has been preprocessed. 'movie.csv' is being imported to this py file for data analysis and to build ML models.
#We first perform sentiment analysis by using TextBlob package to retrieve subjectivity and polarity scores based on review_content column.
#we then use plt package to plot visualizations such as pie chart, histogram, and scatter plot to deliver findings to the audiences.
#The word cloud plot is also being generated to provide insights of which words have the highest frequency in the data, as well as 
#the frequecy distribution visualizations such as bigram and trigram graphs.
#For the machine learning models, we have splitted the dataset into training(80%) and testing(20%) sets. The movie review strings was being
#transform to numerical data by using TF-IDF technique. we then train the dataset into Logistic Regression, Decision Tree, and K-nearest Neighbor models,
#and generate confusion matrix to evaluate model performance for comparison. The ROC curve and the precision-recall curve were also being generated 
#to display the effectiveness of the model and classifier's performance.

__author__ = "Bryan Vega, Terry Hill, Tony Liao, Srividya Panchagnula"
__credits__ = ["Bryan Vega", "Terry Hill", "Tony Liao", "Srividya Panchagnula"]
__version__ = "0.1"
__maintainer__ = "Bryan Vega, Terry Hill, Tony Liao, Srividya Panchagnula"
__email__ = "bvega4@gmu.edu, thill22@gmu.edu, zliao5@gmu.edu, spanchag@gmu.edu"
__status__ = "Prototype"

#-------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import timeit
import time
from collections import Counter

# Natural Language Processing (NLP)
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams
from textblob import TextBlob
from wordcloud import WordCloud

# Model Development and Testing Tools
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer

# Logistic Regression
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.multiclass import OneVsRestClassifier

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# K-Nearest Neighbor (KNN)
from sklearn.neighbors import KNeighborsClassifier



# Record the start time
start_time = time.time()

# read cleaned dataset
df = pd.read_csv("movie.csv")

# In[10]: Sentiment Analysis

# Lexicon sentiment analysis outputs a polarity score of -1 to 1, where -1 represents the highly negative sentiment,
# and 1 shows the highly positive sentiment. A value near 0 represents neutral sentiment.

def sentiment_analysis(df):
    def getSubjectivity(text):
        if isinstance(text, str):
            return TextBlob(text).sentiment.subjectivity
        else:
            return None

    def getPolarity(text):
        if isinstance(text, str):
            return TextBlob(text).sentiment.polarity
        else:
            return None

    df['TextBlob_Subjectivity'] = df['review_content'].apply(getSubjectivity)
    df['TextBlob_Polarity'] = df['review_content'].apply(getPolarity)
    
    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'
    
    df['TextBlob_Analysis'] = df['TextBlob_Polarity'].apply(getAnalysis)
    return df

execution_time = timeit.timeit('sentiment_analysis(df)', globals=globals(), number=1)
hours, rem = divmod(execution_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Sentiment Analysis Completed\nExecution time: {int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}")

# In[11]: Visualizations on Sentiment Analysis

# Sentiment Distribution (Pie chart)
sentiment_counts = df['TextBlob_Analysis'].value_counts()
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
plt.title('Sentiment Distribution')
plt.axis('equal')
plt.show()

# Polarity Distribution (Histogram)
sns.histplot(df['TextBlob_Polarity'], kde=False, bins=30)
plt.title('Polarity Distribution')
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.show()

# Subjectivity Distribution (Histogram)
sns.histplot(df['TextBlob_Subjectivity'], kde=False, bins=30)
plt.title('Subjectivity Distribution')
plt.xlabel('Subjectivity')
plt.ylabel('Count')
plt.show()

# Polarity vs. Subjectivity Scatterplot
sns.scatterplot(x='TextBlob_Polarity', y='TextBlob_Subjectivity', hue='TextBlob_Analysis', data=df)
plt.title('Polarity vs. Subjectivity')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

# Sentiment Analysis over Time (Area Chart)
df['review_date'] = pd.to_datetime(df['review_date'])
df = df.set_index('review_date')
df = df.sort_index()

# Resampling the data (e.g., per week, per month, per quarter, etc.)
# Here, we're resampling by week. Change 'W' to 'M' for month or 'Q' for quarter as needed
resampled_df = df.resample('W')['TextBlob_Analysis'].value_counts().unstack().fillna(0)

# Plotting the area chart
resampled_df.plot.area()
plt.title('Sentiment Analysis over Time (Area Chart)')
plt.xlabel('Time')
plt.ylabel('Count')
plt.show()

# In[20]:

# Resolve NaN values by replacing with an empty string
df['review_content'] = df['review_content'].fillna('')
df['TextBlob_Subjectivity'] = df['TextBlob_Subjectivity'].fillna(0.0)
df['TextBlob_Polarity'] = df['TextBlob_Polarity'].fillna(0.0)

# OTHER TEXT ANALYSIS
# tokenize the review content
def tokenize(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    return [token for token in tokens if token.lower() not in stop_words]

def stem(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Tokenize
df['tokens'] = df['review_content'].apply(tokenize)

# Remove stopwords
df['tokens_no_stopwords'] = df['tokens'].apply(remove_stopwords)

# Stemming
df['stemmed_tokens'] = df['tokens_no_stopwords'].apply(stem)

# Lemmatization
df['lemmatized_tokens'] = df['tokens_no_stopwords'].apply(lemmatize)



# Combine all tokens into a single list
all_tokens = []
for tokens in df['tokens']:
    all_tokens.extend(tokens)

# Create a single string with all tokens
all_tokens_str = ' '.join(all_tokens)

# Generate word cloud
wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(all_tokens_str)

# Plot the word cloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# In[21]: Frequency Distribution Visualization

# Count the frequency of each token
token_counts = Counter(all_tokens)

# Find the 10 most common tokens
most_common_tokens = token_counts.most_common(10)

# Create a DataFrame with the most common tokens
common_tokens_df = pd.DataFrame(most_common_tokens, columns=['Token', 'Count'])

# Plot a bar chart
plt.figure(figsize=(10, 5))
sns.barplot(x='Token', y='Count', data=common_tokens_df)
plt.title('Top 10 Most Frequent Words')
plt.show()



def get_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Generate bigrams (n=2)
df['bigrams'] = df['tokens'].apply(lambda x: get_ngrams(x, 2))

# Combine all bigrams into a single list
all_bigrams = []
for bigram_list in df['bigrams']:
    all_bigrams.extend(bigram_list)

# Count the frequency of each bigram
bigram_counts = Counter(all_bigrams)

# Find the 10 most common bigrams
most_common_bigrams = bigram_counts.most_common(10)

# Create a DataFrame with the most common bigrams
common_bigrams_df = pd.DataFrame(most_common_bigrams, columns=['Bigram', 'Count'])

# Plot a bar chart
plt.figure(figsize=(15, 5))
sns.barplot(x='Bigram', y='Count', data=common_bigrams_df)
plt.title('Top 10 Most Frequent Bigrams')
plt.xticks(rotation=45)
plt.show()

# Generate the frequency distribution
freq_dist = nltk.FreqDist(all_tokens)

plt.figure(figsize=(15, 5))
freq_dist.plot(30, cumulative=False)
plt.show()

plt.figure(figsize=(15, 5))
freq_dist.plot(30, cumulative=True)
plt.show()


# Generate trigrams (n=3)
df['trigrams'] = df['tokens'].apply(lambda x: get_ngrams(x, 3))

# Combine all trigrams into a single list
all_trigrams = []
for trigram_list in df['trigrams']:
    all_trigrams.extend(trigram_list)

# Generate the frequency distribution of trigrams
trigram_freq_dist = nltk.FreqDist(all_trigrams)

# Plot the frequency distribution of trigrams
plt.figure(figsize=(15, 5))
trigram_freq_dist.plot(30, cumulative=False)
plt.show()

# In[25]:


# MACHINE LEARNING MODEL

print('\nLogistic Regression Model - Developing\n')
# LOGISITC REGRESSION MODEL
# Splitting the data into training and testing sets
X = df['review_content']
y = df['TextBlob_Analysis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform the text data using the TF-IDF (Term Frequency - Inverse Document Frequency) vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression Model: Scale the data and increase the max_iter
model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))
model.fit(X_train_tfidf, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Visualize LR Model Results

# Visual 1
# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create a heatmap of the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])

# Set the labels and title
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix Heatmap - Logistic Regression')

# Display the plot
plt.show()

# Visual 2
# Binarize the labels for the multi-class problem
y_test_binarized = label_binarize(y_test, classes=['Negative', 'Neutral', 'Positive'])
y_score = label_binarize(y_pred, classes=['Negative', 'Neutral', 'Positive'])

n_classes = y_test_binarized.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curves
plt.figure()
colors = ['blue', 'orange', 'green']
class_names = ['Negative', 'Neutral', 'Positive']
for i, color, class_name in zip(range(n_classes), colors, class_names):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(class_name, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for Multi-class')
plt.legend(loc="lower right")
plt.show()

# Visual 3
# Binarize the output
y = label_binarize(y, classes=['Negative', 'Neutral', 'Positive'])
n_classes = y.shape[1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model = OneVsRestClassifier(make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000)))
model.fit(X_train_tfidf, y_train)

# Make predictions (probability estimates)
y_score = model.predict_proba(X_test_tfidf)

# Compute Precision-Recall and plot curve
precision = dict()
recall = dict()
average_precision = dict()

# For each class
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

# Plot the micro-averaged Precision-Recall curve
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label='micro-average Precision-recall curve (area = {0:0.2f})'
               ''.format(average_precision["micro"]))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve to multi-class')
plt.legend(loc="lower right")
plt.show()


########################################################################################
# **SKIP** HYPERPARAMETERS FOR LOGISTIC REGRESSION MODEL -- LONG RUNTIME
# param_grid can be adjusted to search for the optimal solver / reduce runtime
#
# # Define the hyperparameters and their possible values
# param_grid_l1 = {
#     'penalty': ['l1'],
#     'C': [0.1, 1, 10],
#     'solver': ['liblinear', 'saga'],
#     'max_iter': [1000, 2000, 3000],
#     'tol': [1e-4, 1e-5, 1e-6]
# }

# param_grid_l2 = {
#     'penalty': ['l2'],
#     'C': [0.1, 1, 10],
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
#     'max_iter': [1000, 2000, 3000],
#     'tol': [1e-4, 1e-5, 1e-6]
# }

# param_grid = [param_grid_l1, param_grid_l2]

# # Initialize the logistic regression model
# logreg = LogisticRegression()

# scaler = MaxAbsScaler()
# X_train_scaled = scaler.fit_transform(X_train_tfidf)  # Remove .toarray() here
# X_test_scaled = scaler.transform(X_test_tfidf)  # Remove .toarray() here

# # Perform grid search with cross-validation
# grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train_scaled, y_train)

# # Get the best hyperparameters
# best_params = grid_search.best_params_
# print("Best hyperparameters:", best_params)

# # Train the logistic regression model with the best hyperparameters
# best_logreg = LogisticRegression(**best_params)
# best_logreg.fit(X_train_scaled, y_train)

# # Make predictions and evaluate the model
# y_pred = best_logreg.predict(X_test_scaled)
# print("Accuracy:", accuracy_score(y_test, y_pred))
########################################################################################
print('\nLogistic Regression Model - Completed\n')


print('\nDecision Tree Model - Developing\n')
# DECISION TREE MODEL
# Define X and y
X = df['review_content']
y = df['TextBlob_Analysis']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize X_train and X_test
vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore', stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Initialize the Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)

# Fit the model
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Print the accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print the classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print the confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visual 1
# Make predictions
y_pred = dt.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Convert confusion matrix to dataframe
cm_df = pd.DataFrame(cm, index=['Negative', 'Neutral', 'Positive'], columns=['Negative', 'Neutral', 'Positive'])

# Plot the heatmap
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
plt.title('Decision Tree \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print('\nDecision Tree Model - Completed\n')


# K-NEAREST NEIGHBOR MODEL
print('\nK-Nearest Neighbor - Developing\n')
# Convert lemmatized tokens back to strings
df['lemmatized_text'] = df['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))

# Split the data into train and test sets
X = df['lemmatized_text']
y = df['TextBlob_Analysis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Convert string labels to numerical labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = knn.predict(X_test_tfidf)

# Print the accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print the classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Print the confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visual 1
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Convert the confusion matrix to a DataFrame
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

# Create a heatmap from the confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for KNN Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print('\nK-Nearest Neighbor - Completed\n')


# Record the end time
end_time = time.time()

# Calculate the time difference and display it
time_elapsed = end_time - start_time
print("Time elapsed: {:.2f} seconds".format(time_elapsed))





















