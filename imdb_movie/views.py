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

#load model 
model = load_model('imdb_movie/training_model/imdbMovie_model.keras')

# Load encoders and scaler
with open('training_model/categorical_columns.pkl', 'rb') as f:
    categorical_columns = pickle.load(f)

with open('training_model/numerical_columns.pkl', 'rb') as f:
    numerical_columns = pickle.load(f)

with open('training_model/standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load và tiền xử lý dữ liệu huấn luyện một lần khi ứng dụng khởi động
training_data = pd.read_csv('imdb_movie/training_model/IMDb_Dataset_2.csv')
training_categorical = training_data[categorical_columns]
training_categorical_encoded = pd.get_dummies(training_categorical, columns=categorical_columns)
# Lưu danh sách các cột đã mã hóa one-hot
training_encoded_columns = training_categorical_encoded.columns



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

        # One-hot encoding các cột categorical
        df_encoded = pd.get_dummies(df, columns=categorical_columns)

        # Bổ sung các cột thiếu để khớp với dữ liệu huấn luyện
        for col in categorical_columns + numerical_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Đảm bảo thứ tự cột đúng
        df_encoded = df_encoded[categorical_columns + numerical_columns]

        # Chuẩn hóa
        X = scaler.transform(df_encoded)

        # Dự đoán
        predicted_rating = model.predict(X)[0][0]
        prediction = round(predicted_rating, 2)

    return render_template("home.html", user=current_user, prediction=prediction)


    return render_template("home.html", prediction=prediction)
