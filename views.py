from flask import render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import forms
import os

model = load_model('training_model/imdbMovie_model.keras')

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

def init_views(app):
    @app.route('/', methods=['GET', 'POST'])
    def index():
        form = forms.MovieForm()
        if form.validate_on_submit():
            rating = predict_rating(form.data)
            return render_template('index.html', form=form, prediction=rating)
        return render_template('index.html', form=form)