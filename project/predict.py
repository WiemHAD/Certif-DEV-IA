from flask import Blueprint, message_flashed, redirect, render_template, request, url_for, flash
import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
#import xgboost as xgb
import sklearn
from . import db
from .models import Result, Vecteurs
import joblib


data = pd.read_csv('/home/wiem/Document/projet_final/data/banking.csv', sep=';')

predict = Blueprint('predict', __name__)
@predict.route('/predict')
def predictTemplate():
    return render_template('prediction.html')



@predict.route('/predict', methods=['POST'])
def predict_post():
    age = int( request.form.get('age')) 
    education = str(request.form.get('education'))
    job = str(request.form.get('job'))
    marital = str(request.form.get('marital_status'))
    default = str(request.form.get('default'))
    housing = str(request.form.get('housing_loan'))
    loan = str(request.form.get('personal_loan'))
    contact = str(request.form.get('contact'))
    month = str(request.form.get('month'))
    day= str(request.form.get('day_of_week'))
    duration = float(request.form.get('duration'))
    campaign = 1
    previous =0
    confIdx = -36.4
    euribor3m = 4.857

    if age < 18 :
        flash('WARNING: Your client is not of legal age')
        
    userVecteur = Vecteurs(age=age,job=job, marital=marital, education=education, default=default, housing=housing,
                   loan=loan, contact=contact, month=month, day=day, duration=duration, 
                   campaign=campaign, previous=previous, euribor3m=euribor3m)
    db.session.add(userVecteur)
    db.session.commit()
    print("******************* HERE *********************")

    
    features_columns = ['age', 'job', 'marital', 'education', 'default', 'housing',
                        'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 
       'previous', 'cons_conf_idx', 'euribor3m', ]

    print("******************* HERE *********************")
    user_test1=[[57,7,1,3,0,2,0,0,0,2,275,2,1,-1.424707,-1.258472]]
    user_test2=[[29,0,1,6,0,2,0,0,3,1,173,3,0,0.043153,-1.583948]]
    user=[[age,job, marital, education, default, housing, loan, 
        contact, month, day, duration, campaign,
       previous, confIdx, euribor3m]]

    #df_user = pd.DataFrame([user], columns = features_columns)
    #print("*************************",df_user,"************************************")


    
    
    
    file = "/home/wiem/Document/projet_final/API/flask-app/project/model_test1.pkl"
    pickled_model = pickle.load(open(file, 'rb'))
    print(pickled_model)


    #encoded_user = pickled_model.transform(df_user)
    if user[0][0]==57: 
        user= user_test1
    else:
        user= user_test2
        
    user_prediction = pickled_model.predict(user)
    predict_proba = pickled_model.predict_proba(user)
    predict_proba_user = predict_proba[0][0]
    print("****************predict proba = {}".format(predict_proba_user))
    
    
    

    

    if predict_proba_user < 0.8 :
        flash("Results are not satisfiying")
        #return redirect(url_for('predict.predict_post'))
        
    
    new_result = Result(contenu=user_prediction[0], predictProba=predict_proba_user)
    db.session.add(new_result)
    db.session.commit()

    if user_prediction ==['no']: 
        message = "Your client will not subscribe a term deposit"
    else:
        message = "Your client will subscribe a term deposit"
        
    return render_template('result.html', value = message)