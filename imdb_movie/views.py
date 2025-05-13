from flask import Blueprint, render_template, request, flash, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import imdb_movie.forms as forms
import os
from flask_login import login_required, current_user
from .models import Note
from . import db
import json

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

















model = load_model('imdb_movie/training_model/imdbMovie_model.keras')
























# Preprocess input data
def preprocess_input(form_data):
    from sklearn.preprocessing import LabelEncoder

    # Initialize encoders (should be fitted during training, here for simplicity)
    certificate_encoder = LabelEncoder()
    certificate_encoder.fit(['G', 'PG', 'PG-13', 'R', 'NC-17', 'Not Rated', 'Approved'])
    genre_encoder = LabelEncoder()
    genre_encoder.fit(['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Horror', 'Mystery'])

    year = (form_data['year'] - 1900) / (2025 - 1900)
    metascore = form_data['metascore'] / 100
    duration = form_data['duration'] / 300

    certificate_one_hot = np.zeros(7)
    certificate_one_hot[certificate_encoder.transform([form_data['certificate']])[0]] = 1

    genre_one_hot = np.zeros(10)
    genre_one_hot[genre_encoder.transform([form_data['genre']])[0]] = 1

    input_data = np.array([year, metascore, duration] + certificate_one_hot.tolist() + genre_one_hot.tolist()).reshape(1, -1)
    return input_data

def predict_rating(form_data):
    input_data = preprocess_input(form_data)
    prediction = model.predict(input_data)
    return np.round(prediction[0][0], 1)
