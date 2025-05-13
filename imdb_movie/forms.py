from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, NumberRange

class MovieForm(FlaskForm):
    
    year = IntegerField('Year', validators=[DataRequired(), NumberRange(min=1900, max=2025)])
    certificate = SelectField('Certificate', choices=[
        ('G'), ('PG'), ('PG-13'), ('R'), ('NC-17'), ('Not Rated'), ('Approved')
    ], validators=[DataRequired()])
    genre = SelectField('Genre', choices=[
        ('Action'), ('Adventure'), ('Animation'), ('Biography'), ('Comedy'), ('Crime'),
        ('Documentary'), ('Drama'), ('Horror'), ('Mystery')
    ], validators=[DataRequired()])
    metascore = IntegerField('MetaScore', validators=[DataRequired(), NumberRange(min=0, max=100)])
    duration = IntegerField('Duration (minutes)', validators=[DataRequired(), NumberRange(min=1)])
    submit = SubmitField('Predict')