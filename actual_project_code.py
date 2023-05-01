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
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
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

# If we want to get a rough idea of the number of words in each df
count_words = lambda x: len(x.split())

Video_Category_df['word_count'] = Video_Category_df['clean_text'].apply(count_words)
Subreddit_Category_df['word_count'] = Subreddit_Category_df['clean_text'].apply(count_words)
Modality_df['word_count'] = Modality_df['clean_text'].apply(count_words)

# sum up the 'word_count' column
total_word_countR = Subreddit_Category_df['word_count'].sum()
total_word_countY = Video_Category_df['word_count'].sum()
total_word_countM = Modality_df['word_count'].sum()

# print the total word count
print('Total word count Reddit:', total_word_countR)# sum up the 'word_count' column
print('Total word count Youtube:', total_word_countY)# sum up the 'word_count' column
print('Total word count Modality:', total_word_countM)# sum up the 'word_count' column

#Split training and test data before going any further to prevent leakage
Youtube_df_train, Youtube_df_test = train_test_split(Video_Category_df, test_size=0.20, stratify=Video_Category_df.kind)
Reddit_df_train, Reddit_df_test = train_test_split(Subreddit_Category_df, test_size=0.20, stratify=Subreddit_Category_df.subreddit)
Modality_df_train, Modality_df_test = train_test_split(Modality_df, test_size=0.20, stratify=Modality_df.modality)

#Start with CountVectorizer
Youtube_CountVec = CountVectorizer(max_features=1500, min_df=5, max_df=0.6,ngram_range=(1, 3), stop_words="english", strip_accents='unicode')
Reddit_CountVec = CountVectorizer(max_features=1500, min_df=5, max_df=0.6,ngram_range=(1, 3), stop_words="english", strip_accents='unicode')
Modality_CountVec = CountVectorizer(max_features=1500, min_df=5, max_df=0.6,ngram_range=(1, 3), stop_words="english", strip_accents='unicode')

#fit Youtube Model
Youtube_text_train = Youtube_CountVec.fit_transform(Youtube_df_train.clean_text)
Youtube_text_test = Youtube_CountVec.transform(Youtube_df_test.clean_text)
Youtube_category_train = Youtube_df_train.kind
Youtube_category_test = Youtube_df_test.kind

YoutubeNBMod = MultinomialNB()
YoutubeNBMod.fit(Youtube_text_train, Youtube_category_train) #fit for training text and classification
predsYoutubeNB = YoutubeNBMod.predict(Youtube_text_test)
print(classification_report(Youtube_category_test, predsYoutubeNB))
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(Youtube_category_test, predsYoutubeNB)))
print('Training set score: {:.4f}'.format(YoutubeNBMod.score(Youtube_text_train, Youtube_category_train)))
print('Test set score: {:.4f}'.format(YoutubeNBMod.score(Youtube_text_test, Youtube_category_test)))
cmYoutube = confusion_matrix(Youtube_category_test, predsYoutubeNB)
print(cmYoutube)
cm_matrixYoutube = pd.DataFrame(data=cmYoutube, columns=['Beauty', 'Cooking', 'Educational', 'Entertainment','Fashion','Fitness','Food','Health','Informative','Law','Videogames'], index=['Beauty', 'Cooking', 'Educational', 'Entertainment','Fashion','Fitness','Food','Health','Informative','Law','Videogames'])
sns.heatmap(cm_matrixYoutube, annot=True, fmt='d', cmap='YlGnBu', annot_kws={"fontsize":8})
plt.gcf().set_size_inches(11, 11) # Adjust the width and height to your needs
plt.savefig('Youtube.pdf')

FP = cm_matrixYoutube.sum(axis=0) - np.diag(cm_matrixYoutube)
FN = cm_matrixYoutube.sum(axis=1) - np.diag(cm_matrixYoutube)
TP = np.diag(cm_matrixYoutube)
TN = cm_matrixYoutube.values.sum() - (FP + FN + TP)

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

# Overall accuracy
print("ACCURACY")
ACC = (TP+TN)/(TP+FP+FN+TN)
print(ACC)

#fit Reddit Model
Reddit_text_train = Reddit_CountVec.fit_transform(Reddit_df_train.clean_text)
Reddit_text_test = Reddit_CountVec.transform(Reddit_df_test.clean_text)
Reddit_subreddit_train = Reddit_df_train.subreddit
Reddit_subreddit_test = Reddit_df_test.subreddit

RedditNBMod = MultinomialNB()
RedditNBMod.fit(Reddit_text_train, Reddit_subreddit_train) #fit for training text and classification
predsRedditNB = RedditNBMod.predict(Reddit_text_test)
print(classification_report(Reddit_subreddit_test, predsRedditNB))
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(Reddit_subreddit_test, predsRedditNB)))
print('Training set score: {:.4f}'.format(RedditNBMod.score(Reddit_text_train, Reddit_subreddit_train)))
print('Test set score: {:.4f}'.format(RedditNBMod.score(Reddit_text_test, Reddit_subreddit_test)))
cmReddit= confusion_matrix(Reddit_subreddit_test, predsRedditNB)
print(cmReddit)
cm_matrixReddit = pd.DataFrame(data=cmReddit, columns=['DIY', 'IAmA', 'LifeProTips', 'NoStupidQuestions','OldSchoolCool','OutOfTheLoop','Showerthoughts','TwoXChromosomes','UIUC','YouShouldKnow','amitheasshole','changemyview','explainlikeimfive','food','gaming','memes','mildlyinfuriating','mildlyinteresting','movies','offmychest','politics','relationship_advice','science','unpopularopinion','worldnews'], index=['DIY', 'IAmA', 'LifeProTips', 'NoStupidQuestions','OldSchoolCool','OutOfTheLoop','Showerthoughts','TwoXChromosomes','UIUC','YouShouldKnow','amitheasshole','changemyview','explainlikeimfive','food','gaming','memes','mildlyinfuriating','mildlyinteresting','movies','offmychest','politics','relationship_advice','science','unpopularopinion','worldnews'])
sns.heatmap(cm_matrixReddit, annot=True, fmt='d', cmap='YlGnBu', annot_kws={"fontsize":8})
plt.gcf().set_size_inches(48, 36) # Adjust the width and height to your needs
plt.savefig('RedditHeatmap.pdf')

FP = cm_matrixReddit.sum(axis=0) - np.diag(cm_matrixReddit)
FN = cm_matrixReddit.sum(axis=1) - np.diag(cm_matrixReddit)
TP = np.diag(cm_matrixReddit)
TN = cm_matrixReddit.values.sum() - (FP + FN + TP)

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

# Overall accuracy
print("ACCURACY")
ACC = (TP+TN)/(TP+FP+FN+TN)
print(ACC)

### Modality ###
Modality_text_train = Modality_CountVec.fit_transform(Modality_df_train.clean_text)
Modality_text_test = Modality_CountVec.transform(Modality_df_test.clean_text)
Modality_type_train = Modality_df_train.modality
Modality_type_test = Modality_df_test.modality

ModalityNBMod = MultinomialNB()
ModalityNBMod.fit(Modality_text_train, Modality_type_train) #fit for training text and classification
predsModalityNB = ModalityNBMod.predict(Modality_text_test)
print(classification_report(Modality_type_test, predsModalityNB))
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(Modality_type_test, predsModalityNB)))
print('Training set score: {:.4f}'.format(ModalityNBMod.score(Modality_text_train, Modality_type_train)))
print('Test set score: {:.4f}'.format(ModalityNBMod.score(Modality_text_test, Modality_type_test)))
cmModality = confusion_matrix(Modality_type_test, predsModalityNB)
cm_matrixModality = pd.DataFrame(data=cmModality, columns=['Spoken','Written'],
                                 index=['Spoken','Written'])
sns.heatmap(cm_matrixModality, annot=True, fmt='d', cmap='YlGnBu', annot_kws={"fontsize":8})
plt.gcf().set_size_inches(48, 36) # Adjust the width and height to your needs
plt.savefig('ModalityHeatmap.pdf')

FP = cm_matrixModality.sum(axis=0) - np.diag(cm_matrixModality)
FN = cm_matrixModality.sum(axis=1) - np.diag(cm_matrixModality)
TP = np.diag(cm_matrixModality)
TN = cm_matrixModality.values.sum() - (FP + FN + TP)

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

# Overall accuracy
print("ACCURACY")
ACC = (TP+TN)/(TP+FP+FN+TN)
print(ACC)


###START OF LSA###

tfidfconverterY = TfidfVectorizer(lowercase = True, max_features=2000, min_df=0.05, max_df=0.6, stop_words=stopwords.words('english'), tokenizer = TOKENIZER.tokenize, ngram_range=(1,3))
tfidfconverterR = TfidfVectorizer(lowercase = True, max_features=2000, min_df=0.05, max_df=0.6, stop_words=stopwords.words('english'), tokenizer = TOKENIZER.tokenize, ngram_range=(1,3))
tfidfconverterM = TfidfVectorizer(lowercase = True, max_features=2000, min_df=0.05, max_df=0.6, stop_words=stopwords.words('english'), tokenizer = TOKENIZER.tokenize, ngram_range=(1,3))

# Fit and Transform the documents
Modality_text_train_LSA = tfidfconverterY.fit_transform(list(Video_Category_df.clean_text))
Reddit_text_train_LSA = tfidfconverterR.fit_transform(list(Subreddit_Category_df.clean_text))
Modality_text_train_LSA = tfidfconverterM.fit_transform(list(Modality_df_train.clean_text))

# Create SVD object
lsa_Reddit = TruncatedSVD(n_components=25, n_iter=100, random_state=42)
lsa_Youtube = TruncatedSVD(n_components=11, n_iter=100, random_state=42)
lsa_Modality = TruncatedSVD(n_components=2, n_iter=100, random_state=42)

# Fit SVD model on data
# Use this instead of fit_transform, apparently doing both there and here puts you at risk for leakage
LSA_Reddit_Out = lsa_Reddit.fit(Reddit_text_train_LSA)
LSA_Youtube_Out = lsa_Youtube.fit(Modality_text_train_LSA)
LSA_Modality_Out = lsa_Modality.fit(Modality_text_train_LSA)

# Print the topics with their terms
print("\n\nREDDIT \n\n")
termsR = tfidfconverterR.get_feature_names_out()

for index, component in enumerate(lsa_Reddit.components_):
    zipped = zip(termsR, component)
    top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:10]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(index)+": ",top_terms_list)

# Print the topics with their terms
print("\n\nYOUTUBE \n\n")
termsY = tfidfconverterY.get_feature_names_out()

for index, component in enumerate(lsa_Youtube.components_):
    zipped = zip(termsY, component)
    top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:10]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(index)+": ",top_terms_list)

# Print the topics with their terms
print("\n\nMODALITY \n\n")
termsM = tfidfconverterM.get_feature_names_out()

for index, component in enumerate(lsa_Modality.components_):
    zipped = zip(termsM, component)
    top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:10]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(index)+": ",top_terms_list)


### START OF LDA ###
### CAN USE GRID SEARCH FOR HYPERPARAMETER TUNING ###

# Define the pipeline
pipelineY = Pipeline([
    ('vectorizer', CountVectorizer(min_df = 0.05, max_df =0.7, max_features=2000,ngram_range=(1, 3), stop_words="english", strip_accents='unicode')),
    ('lda', LatentDirichletAllocation(learning_method='online'))])

# Define the hyperparameter grid to search over
paramsY = {'lda__n_components': [2,4,6,8,10,11,12,14]}

# Perform grid search to find the best hyperparameters
grid_searchY = GridSearchCV(pipelineY, paramsY, cv=5, verbose=1)
grid_searchY.fit(Video_Category_df['clean_text'], Video_Category_df['kind'])

# Print the best hyperparameters
print("Best hyperparameters for Youtube: ", grid_searchY.best_params_)

# Fit the pipeline to the data using the best hyperparameters
pipelineY.set_params(**grid_searchY.best_params_)
pipelineY.fit(Video_Category_df['clean_text'], Video_Category_df['kind'])

# Get the top 10 words for each category
vectorizerY = pipelineY.named_steps['vectorizer']
ldaY = pipelineY.named_steps['lda']
feature_namesY = vectorizerY.get_feature_names_out()
for topic_idx, topic in enumerate(ldaY.components_):
    print("Category %d:" % (topic_idx))
    print(", ".join([feature_namesY[i] for i in topic.argsort()[:-11:-1]]))


# Now for Reddit

PipelineR = Pipeline([
    ('vectorizer', CountVectorizer(min_df = 0.05, max_df =0.7, max_features=2000,ngram_range=(1, 3), stop_words="english", strip_accents='unicode')),
    ('lda', LatentDirichletAllocation(learning_method='online'))])

# Define the hyperparameter grid to search over
paramsR = {'lda__n_components': [3,6,9,12,15,18,21,24,27,30]}

# Perform grid search to find the best hyperparameters
grid_searchR = GridSearchCV(PipelineR, paramsR, cv=5, verbose=1)
grid_searchR.fit(Subreddit_Category_df['clean_text'], Subreddit_Category_df['subreddit'])

# Print the best hyperparameters
print("Best hyperparameters for Reddit: ", grid_searchR.best_params_)

# Fit the pipeline to the data using the best hyperparameters
PipelineR.set_params(**grid_searchR.best_params_)
PipelineR.fit(Subreddit_Category_df['clean_text'], Subreddit_Category_df['subreddit'])

# Get the top 10 words for each category
vectorizerR = RipelineR.named_steps['vectorizer']
ldaR = RipelineR.named_steps['lda']
feature_namesR = vectorizerR.get_feature_names_out()
for topic_idx, topic in enumerate(ldaR.components_):
    print("Category %d:" % (topic_idx))
    print(", ".join([feature_namesR[i] for i in topic.argsort()[:-11:-1]]))

# Define the pipeline for Modality
pipelineM = Pipeline([
    ('vectorizer', CountVectorizer(min_df = 0.05, max_df =0.7, max_features=2000,ngram_range=(1, 3), stop_words="english", strip_accents='unicode')),
    ('lda', LatentDirichletAllocation(learning_method='online'))])

# Define the hyperparameter grid to search over
paramsM = {'lda__n_components': [1,2,3,4]}

# Perform grid search to find the best hyperparameters
grid_searchM = GridSearchCV(pipelineM, paramsM, cv=5, verbose=1)
grid_searchM.fit(Modality_df['clean_text'],Modality_df['modality'])

# Print the best hyperparameters
print("Best hyperparameters for Modality: ", grid_searchM.best_params_)

# Fit the pipeline to the data using the best hyperparameters
pipelineM.set_params(**grid_searchM.best_params_)
pipelineM.fit(Modality_df['clean_text'],Modality_df['modality'])

# Get the top 10 words for each category
vectorizerM = pipelineM.named_steps['vectorizer']
ldaM = pipelineM.named_steps['lda']
feature_namesM = vectorizerM.get_feature_names_out()
for topic_idx, topic in enumerate(ldaM.components_):
    print("Category %d:" % (topic_idx))
    print(" ".join([feature_namesM[i] for i in topic.argsort()[:-11:-1]]))

# dictionary to store words for each topic and number of words per topic to retrive
#https://stackoverflow.com/questions/60790721/topic-modeling-run-lda-in-sklearn-how-to-compute-the-wordcloud
wordsM = {}
n_top_wordsM = 10

for topic, component in enumerate(ldaM.components_):

    # need [::-1] to sort the array in descending order
    indices = np.argsort(component)[::-1][:n_top_wordsM]

    # store the words most relevant to the topic
    wordsM[topic] = [feature_namesM[i] for i in indices]
print(wordsM)

# Perform grid search to find the best hyperparameters
#grid_search = GridSearchCV(pipeline, params, cv=5, n_jobs=-1)
#grid_search.fit(df['text'])

# Get the best LDA model#
#lda_model = grid_search.best_estimator_.named_steps['lda']

# Get the topic distribution for each document
#doc_topic_dist = lda_model.transform(grid_search.best_estimator_.named_steps['tfidf'].transform(df['text']))

# Get the most common words for each topic
tfidf_modelM = grid_searchM.best_estimator_.named_steps['tfidf']
feature_namesM = tfidf_modelM.get_feature_names_out()
top_words = []
for i in range(ldaM.n_components):
    sorted_topicsM = ldaM.components_[i].argsort()[::-1]
    top_words.append([feature_namesM[word] for word in sorted_topicsM[:10]])

# Create a word cloud for each topic
for i in range(ldaM.n_components):
    plt.figure(figsize=(10, 6))
    wcM = WordCloud(background_color='white', width=800, height=400).generate(' '.join(top_words[i]))
    plt.imshow(wcM, interpolation='bilinear')
    plt.axis('off')
    plt.title('Topic {}'.format(i+1))
    plt.savefig('wordcloud_topic_{}.png'.format(i+1))
    plt.clf()
#from
def get_model_topics(model, vectorizer, topics, n_top_words):
    word_dict = {}
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        word_dict[topics[topic_idx]] = top_features
    print(pd.DataFrame(word_dict))
    return pd.DataFrame(word_dict)

get_model_topics(ldaM, vectorizerM, feature_namesM,10)

# If you run it with just 2 components
pipelineM = Pipeline([
    ('vectorizer', CountVectorizer(min_df = 0.05, max_df =0.7, max_features=2000,ngram_range=(1, 3), stop_words="english", strip_accents='unicode')),
    ('lda', LatentDirichletAllocation(learning_method='online', n_components =2))])
pipelineM.fit(Modality_df['clean_text'])

# Get the top 10 words for each category
vectorizerM = pipelineM.named_steps['vectorizer']
ldaM = pipelineM.named_steps['lda']
feature_namesM = vectorizerM.get_feature_names_out()
for topic_idx, topic in enumerate(ldaM.components_):
    print("Category %d:" % (topic_idx))
    print(" ".join([feature_namesM[i] for i in topic.argsort()[:-11:-1]]))

import matplotlib.pyplot as plt
for t in range(ldaM.num_topics):
    plt.figure()
    plt.imshow(WordCloud().fit_words(ldaM.show_topic(t, 200)))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    plt.savefig('Topic'+str(t)+'ModalityLDA.pdf')
plt.clf()
