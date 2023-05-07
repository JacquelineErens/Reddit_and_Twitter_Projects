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
Youtube_df = pd.read_csv('Youtube_transcripts.csv', encoding = 'latin-1') #Dimensions for classification: kind (video category)
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

#simple test to start
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(Modality_df['clean_text'])

# Split data into training and testing sets
X_trainM, X_testM, y_trainM, y_testM = train_test_split(X, Modality_df['modality'], test_size=0.3, random_state=1)

# Make the classifier
classifier = MLPClassifier(hidden_layer_sizes=(10), max_iter=100,activation = 'relu',solver='sgd',random_state=1)

#Fitting the training data to the network
classifier.fit(X_trainM, y_trainM)

#Now working on test set and check predictions
y_predM = classifier.predict(X_testM)
cmNN = confusion_matrix(y_predM, y_testM)

print(classification_report(y_testM, y_predM))
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_testM, y_predM)))
print('Training set score: {:.4f}'.format(classifier.score(X_trainM, y_trainM)))
print('Test set score: {:.4f}'.format(classifier.score(X_testM, y_testM)))
cmMNN= confusion_matrix(y_testM, y_predM)
print(cmMNN)

cmMNN_matrixModality = pd.DataFrame(data=cmMNN, columns=['Spoken','Written'],
                                 index=['Spoken','Written'])
cmMNN_matrixModality_Heatmap = sns.heatmap(cmMNN_matrixModality, annot=True, fmt='d', cmap='YlGnBu', annot_kws={"fontsize": 48})
cmMNN_matrixModality_Heatmap.set_xticklabels(cmMNN_matrixModality_Heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=36)
cmMNN_matrixModality_Heatmap.set_yticklabels(cmMNN_matrixModality_Heatmap.get_yticklabels(), rotation =45, ha = "right",fontsize=36)
plt.gcf().set_size_inches(48, 36) # Adjust the width and height to your needs
plt.savefig('ModalityHeatmapNN.pdf')
plt.clf()

FP = cmMNN_matrixModality.sum(axis=0) - np.diag(cmMNN_matrixModality)
FN = cmMNN_matrixModality.sum(axis=1) - np.diag(cmMNN_matrixModality)
TP = np.diag(cmMNN_matrixModality)
TN = cmMNN_matrixModality.values.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
print("TRUE POSITIVES")
TPR = TP/(TP+FN)
print(TPR)
# Specificity or true negative rate
print("TRUE NEGATIVES")
TNR = TN/(TN+FP)
print(TNR)
# Precision or positive predictive value
print("PRECISION")
PPV = TP/(TP+FP)
print(PPV)
# Negative predictive value
print("NEGATIVE PREDICTIVE")
NPV = TN/(TN+FN)
print(NPV)
# Fall out or false positive rate
print("FALSE POSITIVES")
FPR = FP/(FP+TN)
print(FPR)
# False negative rate
print("FALSE NEGATIVES")
FNR = FN/(TP+FN)
print(FNR)
# False discovery rate
print("FALSE DISCOVERY RATE")
FDR = FP/(TP+FP)
print(FDR)
#Recall
print("RECALL")
RC = TP/(TP+TN)
print(RC)

# Overall accuracy
print("ACCURACY")
ACC = (TP+TN)/(TP+FP+FN+TN)
print(ACC)

# Evaluate the model on the testing set
accuracy = classifier.score(X_testM, y_testM)

print(f'Test Accuracy for Modality: {accuracy}')

### NOW REPEAT FOR YOUTUBE ###

#simple test to start
vectorizerY = CountVectorizer()
XY = vectorizerY.fit_transform(Video_Category_df['clean_text'])

# Split data into training and testing sets
X_trainY, X_testY, y_trainY, y_testY = train_test_split(XY, Video_Category_df['kind'], test_size=0.3, random_state=1)

# Make the classifier
classifierY = MLPClassifier(hidden_layer_sizes=(50), max_iter=100,activation = 'logistic',solver='adam',random_state=1)

#Fitting the training data to the network
classifierY.fit(X_trainY, y_trainY)

#Now working on test set and check predictions
y_predY = classifierY.predict(X_testY)
cmNNY = confusion_matrix(y_predY, y_testY)

print(classification_report(y_testY, y_predY))
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_testY, y_predY)))
print('Training set score: {:.4f}'.format(classifierY.score(X_trainY, y_trainY)))
print('Test set score: {:.4f}'.format(classifierY.score(X_testY, y_testY)))
cmMNNY = confusion_matrix(y_testY, y_predY)
print(cmMNNY)

print(cmMNNY.ravel())

flattened_cm = cmMNNY.ravel()
print(len(flattened_cm))  # Make sure this prints 4

tn1 = cmMNNY.ravel()[0]
fp1 = cmMNNY.ravel()[1]
fn1 = cmMNNY.ravel()[2]
tp1 = cmMNNY.ravel()[3]


print("TP1: ", tp1)

print("FP1: ", fp1)

print("TN1: ", tn1)

print("FN1: ", fn1)

cmMNN_matrixY = pd.DataFrame(data=cmMNNY, columns=['Beauty', 'Cooking', 'Educational', 'Entertainment','Fashion','Fitness','Food','Health','Informative','Law','Videogames'], index=['Beauty', 'Cooking', 'Educational', 'Entertainment','Fashion','Fitness','Food','Health','Informative','Law','Videogames'])
cmMNN_matrixYT_Heatmap = sns.heatmap(cmMNN_matrixY, annot=True, fmt='d', cmap='YlGnBu', annot_kws={"fontsize":48})
cmMNN_matrixYT_Heatmap.set_xticklabels(cmMNN_matrixYT_Heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=36)
cmMNN_matrixYT_Heatmap.set_yticklabels(cmMNN_matrixYT_Heatmap.get_yticklabels(), rotation = 45, ha ="right",fontsize=36)
plt.gcf().set_size_inches(48, 36) # Adjust the width and height to your needs
plt.savefig('YTHeatmapNN.pdf')
plt.clf()

cmMNN_npArray = np.array(cmMNNY)
# Compute TP, FP, FN, and TN from the confusion matrix
n_labels = cmMNN_npArray.shape[0]
flattened_cm = cmMNN_npArray.ravel()
TP = [flattened_cm[(n_labels+1)*i] for i in range(n_labels)]
FP = [np.sum(flattened_cm[n_labels*i:(n_labels+1)*i])-TP[i] for i in range(n_labels)]
FN = [np.sum(flattened_cm[i:n_labels**2:n_labels])-TP[i] for i in range(n_labels)]
TN = [np.sum(flattened_cm)-TP[i]-FP[i]-FN[i] for i in range(n_labels)]

# Print the results
for i in range(n_labels):
    print(f"Class {i}: TP={TP[i]}, FP={FP[i]}, FN={FN[i]}, TN={TN[i]}")

FP = cmMNN_matrixY.sum(axis=0) - np.diag(cmMNN_matrixY)
FN = cmMNN_matrixY.sum(axis=1) - np.diag(cmMNN_matrixY)
TP = np.diag(cmMNN_matrixY)
TN = cmMNN_matrixY.values.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
print("TRUE POSITIVES")
TPR = TP/(TP+FN)
print(TPR)
# Specificity or true negative rate
print("TRUE NEGATIVES")
TNR = TN/(TN+FP)
print(TNR)
# Precision or positive predictive value
print("PRECISION")
PPV = TP/(TP+FP)
print(PPV)
# Negative predictive value
print("NEGATIVE PREDICTIVE")
NPV = TN/(TN+FN)
print(NPV)
# Fall out or false positive rate
print("FALSE POSITIVES")
FPR = FP/(FP+TN)
print(FPR)
# False negative rate
print("FALSE NEGATIVES")
FNR = FN/(TP+FN)
print(FNR)
# False discovery rate
print("FALSE DISCOVERY RATE")
FDR = FP/(TP+FP)
print(FDR)
#Recall
print("RECALL")
RC = TP/(TP+TN)
print(RC)
# Overall accuracy
print("ACCURACY")
ACC = (TP+TN)/(TP+FP+FN+TN)
print(ACC)

# Evaluate the model on the testing set
accuracyY = classifierY.score(X_testY, y_testY)

print(f'Test Accuracy for Youtube: {accuracyY}')

### AND LASTLY REDDIT ###
print("Start of Reddit")
#simple test to start
vectorizerR = CountVectorizer(min_df = 0.05, max_df = 0.70)
XR = vectorizerR.fit_transform(Subreddit_Category_df['clean_text'])

# Split data into training and testing sets
X_trainR, X_testR, y_trainR, y_testR = train_test_split(XR, Subreddit_Category_df['subreddit'], test_size=0.3, random_state=1)

# Make the classifier
classifierR = MLPClassifier(hidden_layer_sizes=(50), max_iter=200,activation = 'logistic',solver='adam',random_state=1)

#Fitting the training data to the network
classifierR.fit(X_trainR, y_trainR)

#Now working on test set and check predictions
y_predR = classifierR.predict(X_testR)
cmNNR = confusion_matrix(y_predR, y_testR)

print(classification_report(y_testR, y_predR))
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_testR, y_predR)))
print('Training set score: {:.4f}'.format(classifierR.score(X_trainR, y_trainR)))
print('Test set score: {:.4f}'.format(classifierR.score(X_testR, y_testR)))
cmMNNR = confusion_matrix(y_testR, y_predR)
print(cmMNNR)

cmMNN_matrixR = pd.DataFrame(data=cmMNNR, columns=['DIY', 'IAmA', 'LifeProTips', 'NoStupidQuestions','OldSchoolCool','OutOfTheLoop','Showerthoughts','TwoXChromosomes','UIUC','YouShouldKnow','amitheasshole','changemyview','explainlikeimfive','food','gaming','memes','mildlyinfuriating','mildlyinteresting','movies','offmychest','politics','relationship_advice','science','unpopularopinion','worldnews'], index=['DIY', 'IAmA', 'LifeProTips', 'NoStupidQuestions','OldSchoolCool','OutOfTheLoop','Showerthoughts','TwoXChromosomes','UIUC','YouShouldKnow','amitheasshole','changemyview','explainlikeimfive','food','gaming','memes','mildlyinfuriating','mildlyinteresting','movies','offmychest','politics','relationship_advice','science','unpopularopinion','worldnews'])
cmMNN_matrixR_Heatmap = sns.heatmap(cmMNN_matrixR, annot=True, fmt='d', cmap='YlGnBu', annot_kws={"fontsize":48})
cmMNN_matrixR_Heatmap.set_xticklabels(cmMNN_matrixR_Heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=36)
cmMNN_matrixR_Heatmap.set_yticklabels(cmMNN_matrixR_Heatmap.get_yticklabels(), rotation=45, ha="right", fontsize=36)
plt.gcf().set_size_inches(48, 36) # Adjust the width and height to your needs
plt.savefig('RedditHeatmapNN.pdf')
plt.clf()

FP = cmMNN_matrixR.sum(axis=0) - np.diag(cmMNN_matrixR)
FN = cmMNN_matrixR.sum(axis=1) - np.diag(cmMNN_matrixR)
TP = np.diag(cmMNN_matrixR)
TN = cmMNN_matrixR.values.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
print("TRUE POSITIVES/RECALL")
TPR = TP/(TP+FN)
print(TPR)
# Specificity or true negative rate
print("TRUE NEGATIVES")
TNR = TN/(TN+FP)
print(TNR)
# Precision or positive predictive value
print("PRECISION")
PPV = TP/(TP+FP)
print(PPV)
# Negative predictive value
print("NEGATIVE PREDICTIVE")
NPV = TN/(TN+FN)
print(NPV)
# Fall out or false positive rate
print("FALSE POSITIVES")
FPR = FP/(FP+TN)
print(FPR)
# False negative rate
print("FALSE NEGATIVES")
FNR = FN/(TP+FN)
print(FNR)
# False discovery rate
print("FALSE DISCOVERY RATE")
FDR = FP/(TP+FP)
print(FDR)
#Recall
print("RECALL")
RC = TP/(TP+FN)
print(RC)
# Overall accuracy
print("ACCURACY")
ACC = (TP+TN)/(TP+FP+FN+TN)
print(ACC)

# Evaluate the model on the testing set
accuracyR = classifierR.score(X_testR, y_testR)

print(f'Test Accuracy for Reddit: {accuracyR}')
