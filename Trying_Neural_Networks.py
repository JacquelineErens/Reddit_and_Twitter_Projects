#Import packages
import pandas as pd
import sys
import re
import string
import joblib
import nltk
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold,cross_val_score,KFold
from sklearn import model_selection, naive_bayes, svm, metrics
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.neural_network import MLPClassifier
#Set seed for consistency
np.random.seed(500)

#Uncomment and run the 3 lines below if you haven't got these packages already
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
TOKENIZER = RegexpTokenizer(r'\b\w{3,}\b')
stop_words = list(set(stopwords.words("english")))
stop_words += list(string.punctuation)
stop_words += ['__', '___']

def change_or_make_path(path_addition):
    if os.path.exists(os.getcwd()+"/"+path_addition):
        os.chdir(os.getcwd()+"/"+path_addition)
    else:
        os.mkdir(path_addition)
        os.chdir(os.getcwd()+"/"+path_addition)

#cleaning functions for processing text
def clean_youtube(input_text):
    text = input_text.replace("\\", " ")
    text = re.sub("\n"," ", input_text) #removes newline characters because for some reason that was still an issue even with the first line
    text = re.sub("[\[].*?[\]]", "", text) #removing everything in brackets, parenthese, or curly brackets
    text = re.sub("-", " ",text)
    text = re.sub("\s{2,}"," ", text) #replace 2 or more spaces with just 1
    text = str(text).lower()
    text = re.sub(r"[0-9]+", '', text)
    #text = " ".join(text.split())
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    lemmText = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(lemmText)
    #print(text)
    return text

def clean_reddit(input_text):

    text = input_text.replace("\\", " ")
    text = re.sub("\n"," ", text)
    text = re.sub("-", " ", text)
    text = re.sub("\s{2,}"," ", text)
    text = str(text).lower()
    text = re.sub(r"[0-9]+", '', text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    lemmText = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(lemmText)
    #print(text)
    return text

# Define a function to convert NLTK part-of-speech tags to WordNet part-of-speech tags
def get_wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wn.ADJ
    elif nltk_tag.startswith('V'):
        return wn.VERB
    elif nltk_tag.startswith('N'):
        return wn.NOUN
    elif nltk_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN  # Default to noun if part-of-speech cannot be determined

# Define a function to lemmatize a sentence
def lemmatize_sentence(sentence):
    # Tokenize the sentence into words
    words = nltk.word_tokenize(sentence)
    # Get the part-of-speech tags for each word
    nltk_tags = nltk.pos_tag(words)
    # Convert the NLTK part-of-speech tags to WordNet part-of-speech tags
    wordnet_tags = [(word, get_wordnet_pos(tag)) for word, tag in nltk_tags]
    # Lemmatize each word using its corresponding part-of-speech tag
    lemmatized_words = [lemmatizer.lemmatize(word, tag) for word, tag in wordnet_tags]
    # Join the lemmatized words back into a sentence
    lemmatized_sentence = ' '.join(lemmatized_words)
    return lemmatized_sentence

#Load in datasets
Youtube_df = pd.read_csv('Youtube_transcripts.csv', encoding = 'latin-1') #Dimensions for classification: kind (video category), author
Reddit_df = pd.read_csv('Reddit_data4.csv', encoding = 'latin-1') #Dimensions for categorization: subreddit

### Clean Datasets ###

#Drop empty rows or youtube videos where you couldn't extract the transcripts
Reddit_df.dropna(subset=['cleaned'], inplace=True)
Youtube_df.dropna(subset=['text'], inplace=True)
Youtube_df = Youtube_df[Youtube_df['text'] != "Error: Unable to extract transcript."]

#Apply our cleaning functions (see above)
Reddit_df['clean_text'] = Reddit_df['cleaned'].apply(clean_reddit)
Youtube_df['clean_text'] = Youtube_df['text'].apply(clean_youtube)
Youtube_df['lemma_text'] = Youtube_df['clean_text'].apply(lambda x: ' '.join([lemmatize_sentence(sentence) for sentence in nltk.sent_tokenize(x)]))
Reddit_df['lemma_text'] = Reddit_df['clean_text'].apply(lambda x: ' '.join([lemmatize_sentence(sentence) for sentence in nltk.sent_tokenize(x)]))
#Reddit_df.dropna(subset=['clean_text'], inplace=True)
#Youtube_df.dropna(subset=['clean_text'], inplace=True)
#Reddit_df.rename(columns={'lemma_text': 'clean_text'}, inplace=True)
#Youtube_df.rename(columns={'lemma_text': 'clean_text'}, inplace=True)

#create a separate dataset with just text and modality in case we have time to do this
Youtube_df['modality'] = 'spoken'
Reddit_df['modality'] = 'written'

print(Reddit_df.head())
print(Youtube_df.head())

#after the first time of doing train/test split, some categories in the youtube dataset had too few members, so for the purpose of this project they will be dropped
Youtube_df = Youtube_df[Youtube_df['kind'] != "Q&A"]
Youtube_df = Youtube_df[Youtube_df['kind'] != "Learning"]
Youtube_df = Youtube_df[Youtube_df['kind'] != "DIY"]
Youtube_df = Youtube_df[Youtube_df['kind'] != "Family"]

#Create a dataframe of all of them split by modality
Youtube_df_Modality=Youtube_df[['clean_text','modality']]
Reddit_df_Modality=Reddit_df[['clean_text','modality']]

#create the 3 datasets in their final form, without all the other unneeded columns
Modality_df = pd.concat([Youtube_df_Modality, Reddit_df_Modality], ignore_index=True, axis=0)
Video_Category_df = Youtube_df[['clean_text','kind']]
Subreddit_Category_df = Reddit_df[['clean_text','subreddit']]

#Save them to csv so if we want to do more we can write code to come after this cleaning
Modality_df.to_csv('Modality_df.csv')
Video_Category_df.to_csv('Video_Category_df.csv')
Subreddit_Category_df.to_csv('Subreddit_Category_df.csv')

### NEURAL NETWORK ON MODAYLITY ###
# Split data into training and testing sets
X_trainM, X_testM, y_trainM, y_testM = train_test_split(Modality_df['clean_text'], Modality_df['modality'], test_size=0.3, random_state=543)

# Create a pipeline that converts text to a bag-of-words representation and fits an MLPClassifier
pipelineM = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MLPClassifier())
])

# Set up the grid search to find the optimal hyperparameters
parametersM = {
    'classifier__hidden_layer_sizes': [(10,), (50,), (100,), (50,50), (100,100)],
    'classifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'classifier__solver': ['lbfgs', 'sgd', 'adam'],
    'classifier__max_iter': [500, 1000]
}

print("Starting Grid Search")
#run the grid search, using all available cores
grid_searchM = GridSearchCV(pipelineM, parametersM, cv=5, n_jobs=-1)

#Run grid search on the training data
grid_searchM.fit(X_trainM, y_trainM)

print("Best hyperparameters:", grid_searchM.best_params_)
print("Best accuracy: ", grid_searchM.best_score_)

# Fit the pipeline with the optimal hyperparameters on the training data
pipelineM.set_params(**grid_searchM.best_params_)
pipelineM.fit(X_trainM, y_trainM)

# Predict on the test data
y_predM = pipelineM.predict(X_testM)

# Calculate accuracy
accuracyM = accuracy_score(y_testM, y_predM)
print("Accuracy: ", accuracyM)

### NOW REPEAT FOR YOUTUBE ###

# Split data into training and testing sets
X_trainY, X_testY, y_trainY, y_testY = train_test_split(Video_Category_df['clean_text'], Video_Category_df['kind'], test_size=0.3, random_state=543)

# Create a pipeline that converts text to a bag-of-words representation and fits an MLPClassifier
pipelineY = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MLPClassifier())
])

# Set up the grid search to find the optimal hyperparameters
parametersY = {
    'classifier__hidden_layer_sizes': [(10,), (50,), (100,), (50,50), (100,100)],
    'classifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'classifier__solver': ['lbfgs', 'sgd', 'adam'],
    'classifier__max_iter': [500, 1000]
}

print("Starting Grid Search")
#run the grid search, using all available cores
grid_searchY = GridSearchCV(pipelineY, parametersY, cv=5, n_jobs=-1)

#Run grid search on the training data
grid_searchY.fit(X_trainY, y_trainY)

print("Best hyperparameters for youtube:", grid_searchY.best_params_)
print("Best accuracy: ", grid_searchY.best_score_)

# Fit the pipeline with the optimal hyperparameters on the training data
pipelineY.set_params(**grid_searchY.best_params_)
pipelineY.fit(X_trainY, y_trainY)

# Predict on the test data
y_predY = pipelineY.predict(X_testY)

# Calculate accuracy
accuracyY = accuracy_score(y_testY, y_predY)
print("Accuracy: ", accuracyY)


### NOW REPEAT FOR REDDIT ###

# Split data into training and testing sets
X_trainR, X_testR, y_trainR, y_testR = train_test_split(Subreddit_Category_df['clean_text'], Subreddit_Category_df['subreddit'], test_size=0.3, random_state=543)

# Create a pipeline that converts text to a bag-of-words representation and fits an MLPClassifier
pipelineR = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MLPClassifier())
])

# Set up the grid search to find the optimal hyperparameters
parametersR = {
    'classifier__hidden_layer_sizes': [(10,), (50,), (100,), (50,50), (100,100)],
    'classifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'classifier__solver': ['lbfgs', 'sgd', 'adam'],
    'classifier__max_iter': [500, 1000]
}

print("Starting Grid Search on Reddit")
#run the grid search, using all available cores
grid_searchR = GridSearchCV(pipelineR, parametersR, cv=5, n_jobs=-1)

#Run grid search on the training data
grid_searchR.fit(X_trainR, y_trainR)

print("Best hyperparameters for reddit:", grid_searchR.best_params_)
print("Best accuracy: ", grid_searchR.best_score_)

# Fit the pipeline with the optimal hyperparameters on the training data
pipelineR.set_params(**grid_searchR.best_params_)
pipelineR.fit(X_trainR, y_trainR)

# Predict on the test data
y_predR = pipelineR.predict(X_testR)

# Calculate accuracy
accuracyR = accuracy_score(y_testR, y_predR)
print("Accuracy: ", accuracyR)
