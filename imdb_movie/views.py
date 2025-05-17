from flask import Blueprint, render_template, request, flash, jsonify 
from tensorflow.keras.models import load_model
import numpy as np
import imdb_movie.forms as forms
import os
from flask_login import login_required, current_user
from .models import Note
from . import db
import json
import pickle
import pandas as pd
import joblib

views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST': 
        note = request.form.get('note')#Gets the note from the HTML 

        if len(note) < 1:
            flash('Note is too short!', category='error') 
        else:
            new_note = Note(data=note, user_id=current_user.id)  #providing the schema for the note 
            db.session.add(new_note) #adding the note to the database 
            db.session.commit()
            flash('Note added!', category='success')

    return render_template("home.html", user=current_user)


@views.route('/delete-note', methods=['POST'])
def delete_note():  
    note = json.loads(request.data) # this function expects a JSON from the INDEX.js file 
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({})

#--------------------------------------------model prediction--------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tải mô hình và các đối tượng đã lưu
model = load_model(os.path.join(BASE_DIR, 'training_model', 'imdbMovie_model.keras'))
scaler = joblib.load(os.path.join(BASE_DIR, 'training_model', 'standard_scaler.pkl'))
cat_columns = joblib.load(os.path.join(BASE_DIR, 'training_model', 'categorical_columns.pkl'))
num_columns = joblib.load(os.path.join(BASE_DIR, 'training_model', 'numerical_columns.pkl'))


def preprocess_input(df):
    # Phân chia cột số và cột phân loại
    df_num = df[num_columns]
    df_cat = df.drop(columns=num_columns)

    # Chuẩn hóa dữ liệu số
    df_num_scaled = pd.DataFrame(scaler.transform(df_num), columns=num_columns)

    # One-hot encode dữ liệu phân loại
    df_cat_encoded = pd.get_dummies(df_cat)
    
    # Đảm bảo các cột one-hot giống với training
    for col in cat_columns:
        if col not in df_cat_encoded:
            df_cat_encoded[col] = 0
    df_cat_encoded = df_cat_encoded[cat_columns]  # đúng thứ tự

    # Kết hợp lại
    final_input = pd.concat([df_num_scaled, df_cat_encoded], axis=1)
    return final_input

@views.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        year = int(request.form['year'])
        certificate = request.form['certificates']
        genre = request.form['genre']
        director = request.form['director']
        star_cast = request.form['star_cast']
        metascore = int(request.form['metascore'])
        duration = int(request.form['duration'])

        # Tạo DataFrame input
        input_dict = {
            'Year': [year],
            'Certificate': [certificate],
            'Genre': [genre],
            'Director': [director],
            'Star': [star_cast],
            'Meta_score': [metascore],
            'Duration': [duration]
        }
        df = pd.DataFrame(input_dict)

        #Tiền xử lý dữ liệu , cho dữ liệu từ form khớp với dữ liệu đã train
        input_processed = preprocess_input(df)
        # Dự đoán
        pred = model.predict(input_processed)[0][0]
        prediction = round(pred, 2)


    return render_template("home.html", user=current_user, prediction=prediction)


