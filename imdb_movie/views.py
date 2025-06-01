from flask import Blueprint, render_template, request
import numpy as np
import os
import joblib
import pandas as pd

views = Blueprint('views', __name__, template_folder='templates')

# Lấy đường dẫn thư mục hiện tại
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load mô hình Random Forest
model_path = os.path.join(BASE_DIR, 'training_model', 'model.pkl')
model = joblib.load(model_path)

# Danh sách các thể loại theo đúng thứ tự khi huấn luyện
genres_path = os.path.join(BASE_DIR, 'training_model', 'genres.pkl')
genres = joblib.load(genres_path)


@views.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@views.route('/predict', methods=['POST'])
def predict():
    try:
        year = int(request.form['year'])
        duration = int(request.form['duration'])
        votes = int(request.form['votes'])
        director = request.form['directorName']
        writer = request.form['writeName']
        selected_genres = request.form.getlist('genres')  

        input_dict = {
            'directorName': [director],
            'writeName': [writer],
            'runtime': [duration],
            'year': [year],
            'votes': [votes],  # nếu có votes trong model thì thêm vào đây
        }

        for g in genres:
            input_dict[g] = [1 if g in selected_genres else 0]

        input_df = pd.DataFrame(input_dict)

        pred = model.predict(input_df)[0]
        prediction = round(pred, 2)
        return render_template('index.html',prediction=prediction,year=year,votes=votes,duration=duration,directorName=director,writeName=writer,selected_genres=selected_genres)

    except Exception as e:
        print("Error in prediction:", e)
        return render_template('index.html', prediction="Lỗi trong quá trình dự đoán")