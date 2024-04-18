from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np
import pandas as pd
from PIL import Image
#from tensorflow.keras.models import load_model


app = Flask(__name__)

def predict2(values, dic):
    model = pd.read_pickle('models/kidney.pkl')
    values = np.asarray(values)
    return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/working")
def workingPage():
    return render_template('working.html')

@app.route("/graph")
def graph():
    return render_template('graph.html')


def prediction(l):
    import pickle

    # Load the model from the pickle file
    filename = 'models/my_classifier.pkl'
    with open(filename, 'rb') as file:
        clf = pickle.load(file)

    # Define your single data point as a list
    single_data_point = l # 

    # Now you can use the loaded model to make predictions on this single data point
    prediction = clf.predict([single_data_point])
    probabilities = clf.predict_proba([single_data_point])

    # Print or use the prediction and probabilities as needed
    return max(prediction),max(probabilities)



@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():

    if request.method == 'POST':
        age = float(request.form['age'])
        bp = float(request.form['bp'])
        al = float(request.form['al'])
        su = float(request.form['su'])
        rbc = float(request.form['rbc'])
        pc = float(request.form['pc'])
        pcc = float(request.form['pcc'])
        ba = float(request.form['ba'])
        bgr = float(request.form['bgr'])
        bu = float(request.form['bu'])
        sc = float(request.form['sc'])
        pot = float(request.form['pot'])
        wc = float(request.form['wc'])
        htn = float(request.form['htn'])
        dm = float(request.form['dm'])
        cad = float(request.form['cad'])
        pe = float(request.form['pe'])
        ane = float(request.form['ane'])
        sg = float(request.form['sg'])
        sodium = float(request.form['sodium'])
        hg = float(request.form['hg'])
        pcv = float(request.form['pcv'])
        rbc_count = float(request.form['rbcc'])
        appetite = float(request.form['apetite'])
        data = [age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc,sodium, pot,hg,pcv, wc,rbc_count, htn, dm, cad,appetite, pe, ane]
        print(data)
        pred,prob=prediction(data)
    

    return render_template('predict.html', pred = pred,prob=prob,data=data)


if __name__ == '__main__':
	app.run(debug = True)
