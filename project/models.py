from argparse import _CountAction
from asyncio import constants
from flask_login import UserMixin
from . import db

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))
    
    
    
    
class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True) 
    contenu = db.Column(db.String(100))
    predictProba=db.Column(db.Float(10))
    

class Vecteurs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age=db.Column(db.Integer)
    job=db.Column(db.String(100))
    marital =db.Column(db.String(10))
    education=db.Column(db.String(10))
    default=db.Column(db.String(10))
    housing=db.Column(db.String(10))
    loan=db.Column(db.String(10))
    contact=db.Column(db.String(10))
    month=db.Column(db.String(10))
    day=db.Column(db.String(10))
    duration=db.Column(db.Float(10))   
    campaign=db.Column(db.String(10))
    previous=db.Column(db.Float(10))
    poutcome=db.Column(db.String(10))
    confIdx=db.Column(db.Float(10))
    euribor3m=db.Column(db.Float(10))



