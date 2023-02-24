import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow.compat.v2 as tf
from keras import models
from keras.engine.functional import Functional
from keras.engine.sequential import Sequential
from keras.engine.training import Model
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout,Bidirectional,GRU
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from itertools import chain
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
import nltk
import pickle

df = pd.read_json(r'Sarcasm_Headlines_Dataset.json', lines=True)

print(df.head())

df.isnull().sum()

df['website'] = df.article_link.apply(lambda x: x.split('/')[2])
print(df.pivot_table(index = 'website', columns = 'is_sarcastic', values = 'article_link', aggfunc='count').fillna(0).astype(int))

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))

print(stop_words)

df['headline'] = df['headline'].apply(lambda x: ' '.join([item for item in str(x).split() if item not in stop_words]))

df['headline'] = df['headline'].apply(lambda x: x.lower())
df['headline'] = df['headline'].str.replace('[^\w\s]','')
df['headline'] = df['headline'].str.replace('[0-9]+', '')
common_top10 = pd.Series(' '.join(df['headline']).split()).value_counts()[:10]
print(common_top10)

sentences = df['headline'].values.tolist()
target = df['is_sarcastic'].values.tolist()

num_words = 1000
tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(sentences)
tokens = tokenizer.texts_to_sequences(sentences)

numTokens = [len(token) for token in tokens]
numTokens = np.array(numTokens)
print("Tokens'mean",np.mean(numTokens))
print("Max", np.max(numTokens))
print("Argmax", np.argmax(numTokens))

X = tokenizer.texts_to_sequences(df['headline'].values)
X = pad_sequences(X)
y = pd.get_dummies(df['is_sarcastic']).values

max_tokens = int(np.mean(numTokens) + 2*np.std(numTokens))
padding_data = pad_sequences(tokens, maxlen=max_tokens)
print(padding_data.shape)
print(len(target))


x = df["headline"]
y = df["is_sarcastic"]

train_x, test_x, train_y, test_y = model_selection.train_test_split(x, y,
                                                                    test_size = 0.20,
                                                                    shuffle = True,
                                                                    random_state = 11)

tf_idf_word_vectorizer = TfidfVectorizer(analyzer = "word")
tf_idf_word_vectorizer.fit(train_x)

x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

x_train_tf_idf_word.toarray()

log = linear_model.LogisticRegression()
log_model = log.fit(x_train_tf_idf_word, train_y)
accuracy = model_selection.cross_val_score(log_model,
                                           x_test_tf_idf_word,
                                           test_y,
                                           cv = 20).mean()
pickle.dump(log_model, open("model.pkl","wb"))
print(x_test_tf_idf_word)

print("\nLogistic regression model with 'tf-idf' method")
print("Accuracy ratio: ", accuracy)



