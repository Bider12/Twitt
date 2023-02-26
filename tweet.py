import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
from sklearn import model_selection, preprocessing, linear_model, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import ensemble
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from termcolor import colored
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import nltk


data = pd.read_csv("data.csv")

print(data.head())
print(data.isnull().sum())
print(data.shape)

# 21/4009 = 0.5% missing data, we can delete
data.dropna()
data['website'] = data.URLs.apply(lambda x: x.split('/')[2])
print(data.pivot_table(index = 'website', columns = 'Label', values = 'URLs', aggfunc='count').fillna(0).astype(int))

data['text'] = data['Headline'] + " " + data['Body']
data = data.drop(columns = ['Headline', 'Body','URLs'])

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

# NLP libraries to clean the text data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

stop_words = set(stopwords.words('english'))

ps = PorterStemmer()
data['text'] = data['text'].str.replace('[^\w\s]', ' ')
data['text'] = data['text'].apply(lambda x: ' '.join([item for item in str(x).split() if item not in stop_words]))
data['text'] = data['text'].apply(lambda x: x.lower())



print(data['text'][1])

X = data['text']
Y = data['Label']

#Split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

#Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
print(xv_test)

#1. Logistic Regression - used because this model is best suited for binary classification
LR_model = LogisticRegression()

#Fitting training set to the model
LR_model.fit(xv_train,y_train)

#Predicting the test set results based on the model
lr_y_pred = LR_model.predict(xv_test)

#Calculate the accurracy of this model
score = accuracy_score(y_test,lr_y_pred)
print('Accuracy of LR model is ', score)

#2. Support Vector Machine(SVM) - SVM works relatively well when there is a clear margin of separation between classes.
svm_model = SVC(kernel='linear')

#Fitting training set to the model
svm_model.fit(xv_train,y_train)

#Predicting the test set results based on the model
svm_y_pred = svm_model.predict(xv_test)

#Calculate the accuracy score of this model
score = accuracy_score(y_test,svm_y_pred)
print('Accuracy of SVM model is ', score)

#3. Random Forest Classifier
RFC_model = RandomForestClassifier(random_state=0)

#Fitting training set to the model
RFC_model.fit(xv_train, y_train)

#Predicting the test set results based on the model
rfc_y_pred = RFC_model.predict(xv_test)

#Calculate the accuracy score of this model
score = accuracy_score(y_test,rfc_y_pred)
print('Accuracy of RFC model is ', score)

model = svm_model.fit(xv_train,y_train)
print(model)
print(xv_test)
pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorization, open("vector.pkl","wb"))


