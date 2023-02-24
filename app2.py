from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np
import pickle

#Create Flask app
app = Flask(__name__)

#load the pickle model
model = pickle.load(open("model.pkl","rb"))
print(model.predict)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
    return render_template('result.html', predict=model(data))
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)

