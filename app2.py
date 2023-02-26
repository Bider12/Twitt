from flask import Flask, request, render_template, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import pandas as pd
import numpy as np
from numpy import reshape
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords



#Create Flask app


app = Flask(__name__)

#load the pickle model
model = pickle.load(open("model.pkl","rb"))
vectorization = pickle.load(open("vector.pkl",'rb'))
#data = pd.Series(["Trump is king"])
#vectorization = TfidfVectorizer()

def wordopt(text):
    ps = PorterStemmer()
    pattern = re.compile('[^a-zA-Z]')
    text = re.sub(pattern, ' ',str(text))
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text
#standaryzacja tekstu
#zamiana w dataframe





#xv_test = vectorization.transform(data)

#print(model.predict(data))
#
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        #message = "Trump is king"
        message = {'text':[message]}
        df = pd.DataFrame(message)
        df['text'] = df['text'].apply(wordopt)
        message = df['text']
        vectorized = vectorization.transform(message)
        #prediction = model(vectorized)


        return render_template('result.html', predict=model.predict(vectorized))
    return render_template("home.html")
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)

