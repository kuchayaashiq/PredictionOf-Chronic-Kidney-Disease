from flask import Flask, request, url_for, render_template
from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from pywebio.input import *
from pywebio.output import *
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
app=Flask(__name__)
model=pickle.load(open('model.pkl', 'rb'))
@app.route('/', methods=['GET'])
def home():
   return render_template('index.html')

standard_to = StandardScaler()
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        WC = float(request.form['wc'])
        BGR = float(request.form['bgr'])
        BU = float(request.form['bu'])
        SC = float(request.form['sc'])
        PCV = float(request.form['pcv'])
        AL = float(request.form['al'])
        HEMO = float(request.form['hemo'])
        AGE = float(request.form['age'])
        SU = float(request.form['su'])
        HTN = float(request.form['htn'])

        prediction=model.predict([[WC,BGR,BU,SC,PCV,AL,HEMO,AGE,SU,HTN]])

        if prediction==1:
            return render_template('index.html',pred='Patient has Chronic Kidney Disease. ')
        else:
            return render_template('index.html',pred='Patient  is safe Patient doesnot have CKD')
    else:
        return render_template('index.html')
    
# if __name__=='__main__':
#      app.run(debug=True)

app.add_url_rule('/tool', 'webio_view', webio_view(predict), methods=['GET','POST', 'OPTIONS'])

app.run(host='localhost', port=80)