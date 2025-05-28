from flask import Flask
from imdb_movie.views import views  # import blueprint

def create_app():
    app = Flask(__name__)
    app.register_blueprint(views, url_prefix='/')    # đăng ký blueprint
    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
