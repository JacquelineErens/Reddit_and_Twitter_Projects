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
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold,cross_val_score,KFold
from sklearn import model_selection, naive_bayes, svm, metrics
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#Set seed for consistency
np.random.seed(500)

#Uncomment and run the 3 lines below if you haven't got these packages already
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
TOKENIZER = RegexpTokenizer(r'\b\w{3,}\b')
stop_words = list(set(stopwords.words("english")))
stop_words += list(string.punctuation)
stop_words += ['__', '___']


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

Modality_df.to_csv('Modality_df.csv')
Video_Category_df.to_csv('Video_Category_df.csv')
Subreddit_Category_df.to_csv('Subreddit_Category_df.csv')

#Split training and test data before going any further to prevent leakage
R_text_train, R_text_test, R_subreddit_train, R_subreddit_test = train_test_split(Video_Category_df['clean_text'],Video_Category_df['kind'], test_size=0.20)
Y_text_train, Y_text_test, Y_kind_train, Y_kind_test = train_test_split(Subreddit_Category_df['clean_text'],Subreddit_Category_df['subreddit'], test_size=0.20)
M_text_train, M_text_test, M_modality_train, M_modality_test = train_test_split(Modality_df['clean_text'],Modality_df['modality'], test_size=0.20)

# Define the vectorizer to use
vec = TfidfVectorizer(stop_words='english', max_features = 2000, min_df =0.05)

# Define the clustering model to use
model = AgglomerativeClustering()

# Define the pipeline to use for hyperparameter tuning
pipeline = Pipeline([
    ('vec', vectorizer),
    ('model', model)
])

# set hyperparameters to be tuned
parameters = {
    'vec__max_df': (0.5, 0.6),
    'model__n_clusters': [1,2, 3, 4],
    'model__linkage': ['ward', 'complete', 'average']
}


# perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(pipeline, parameters, cv=5) #5 is the default
grid_search.fit(Video_Category_df['clean_text'])

# extract the best model from the grid search
best_model = grid_search.best_estimator_

# fit the best model to the data
best_model.fit(Video_Category_df['clean_text'])

# Compute the linkage matrix and plot the dendrogram
linkage_matrix = linkage(best_model.named_steps['vec'].transform(Video_Category_df['clean_text']).todense(), method='ward')
plt.figure(figsize=(22, 22))
dendrogram(linkage_matrix)
plt.savefig('YoutubeLinkageMatrix.pdf')
plt.show()
