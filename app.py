from flask import Flask, render_template
import views


app = Flask(__name__)
app.secret_key = '123'

views.init_views(app)

if __name__ == '__main__':
    app.run(debug=True)