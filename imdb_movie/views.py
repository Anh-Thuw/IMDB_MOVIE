from flask import Blueprint, render_template, request
import numpy as np
import os
import joblib

views = Blueprint('views', __name__, template_folder='templates')

# Lấy đường dẫn thư mục hiện tại
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load mô hình Random Forest
model_path = os.path.join(BASE_DIR, 'training_model', 'random_forest_model.pkl')
model = joblib.load(model_path)

# Danh sách các thể loại theo đúng thứ tự khi huấn luyện
genres = ['Mystery', 'Drama', 'Musical', 'Fantasy', 'Adventure', 'Western', 'Thriller', 'War',
          'Biography', 'Family', 'Sport', 'Film-Noir', 'Music', 'Sci-Fi', 'Animation', 'Romance',
          'Crime', 'Action', 'Comedy', 'History']


@views.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@views.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu số
        year = int(request.form['year'])
        duration = int(request.form['duration'])
        metascore = int(request.form['metascore'])

        # Xử lý votes và gross có thể để trống
        votes = request.form.get('votes')
        gross = request.form.get('gross')

        votes = int(votes) if votes else np.nan
        gross = float(gross) if gross else np.nan

        # Lấy danh sách genre đã chọn và tạo vector 0/1
        selected_genres = request.form.getlist('genres')
        genre_vector = [1 if genre in selected_genres else 0 for genre in genres]

        # Tạo mảng đầu vào cho mô hình
        input_data = np.array([[year, duration, votes, metascore, gross] + genre_vector])

        # Dự đoán điểm IMDb
        prediction = model.predict(input_data)
        predicted_rating = round(float(prediction[0]), 2)

        return render_template("index.html", prediction=predicted_rating,
                                                year=year,
                                                duration=duration,
                                                metascore=metascore,
                                                votes=votes,
                                                gross=gross,
                                                selected_genres=selected_genres)

    except Exception as e:
        print("Error in prediction:", e)
        return render_template("index.html", prediction="Lỗi trong quá trình dự đoán")