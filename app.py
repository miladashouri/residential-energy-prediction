from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange
import numpy as np  
import joblib



def return_prediction(rf_random_1,rf_random_2,scaler_x,scaler_y1,scaler_y2,sample_json):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you

    Relative_compactness=sample_json['Relative_compactness']
    Surface_area=sample_json['Surface_area']
    Wall_area=sample_json['Wall_area']
    Roof_area=sample_json['Roof_area']
    Overall_height=sample_json['Overall_height']
    Orientation=sample_json['Orientation']
    Glazing_area=sample_json['Glazing_area']
    Glazing_area_distribution=sample_json['Glazing_area_distribution']




    
    sample = [[Relative_compactness,Surface_area,Wall_area,Roof_area,Overall_height,Orientation,Glazing_area,Glazing_area_distribution]]
    sample = scaler_x.transform(sample)
    
    
    Heating_Load = rf_random_1.predict(sample)
    Heating_Load=scaler_y1.inverse_transform(Heating_Load.reshape(-1, 1))
    
    Cooling_Load= rf_random_2.predict(sample)
    Cooling_Load=scaler_y2.inverse_transform(Cooling_Load.reshape(-1, 1))
    
    return Heating_Load,Cooling_Load



app = Flask(__name__)
# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'mysecretkey'


# LOAD THE MODEL AND THE SCALERS
rf_random_1 = joblib.load("./models/rf_random_1.joblib")
rf_random_2 = joblib.load("./models/rf_random_2.joblib")
scaler_x = joblib.load("./models/scaler_x.pkl")
scaler_y1 = joblib.load("./models/scaler_y1.pkl")
scaler_y2 = joblib.load("./models/scaler_y2.pkl")



# WTForm Class
# Lots of fields available: http://wtforms.readthedocs.io/en/stable/fields.html
class FlowerForm(FlaskForm):
    
    Relative_compactness = TextField('Relative compactness')
    Surface_area = TextField('Surface area')
    Wall_area = TextField('Wall area')
    Roof_area = TextField('Roof area')
    Overall_height = TextField('Overall height')
    Orientation = TextField('Orientation')
    Glazing_area = TextField('Glazing area')
    Glazing_area_distribution = TextField('Glazing area distribution')

    submit = SubmitField('Predict')



@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = FlowerForm()
    # If the form is valid on submission
    if form.validate_on_submit():
        # Grab the data from the form

        session['Relative_compactness'] = form.Relative_compactness.data
        session['Surface_area'] = form.Surface_area.data
        session['Wall_area'] = form.Wall_area.data
        session['Roof_area'] = form.Roof_area.data
        session['Overall_height'] = form.Overall_height.data
        session['Orientation'] = form.Orientation.data
        session['Glazing_area'] = form.Glazing_area.data
        session['Glazing_area_distribution'] = form.Glazing_area_distribution.data

        return redirect(url_for("prediction"))


    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['Relative_compactness'] = float(session['Relative_compactness'])
    content['Surface_area'] = float(session['Surface_area'])
    content['Wall_area'] = float(session['Wall_area'])
    content['Roof_area'] = float(session['Roof_area'])
    content['Overall_height'] = float(session['Overall_height'])
    content['Orientation'] = float(session['Orientation'])
    content['Glazing_area'] = float(session['Glazing_area'])
    content['Glazing_area_distribution'] = float(session['Glazing_area_distribution'])

    results = return_prediction(rf_random_1=rf_random_1,rf_random_2=rf_random_2,scaler_x=scaler_x,scaler_y1=scaler_y1,scaler_y2=scaler_y2,sample_json=content)
    

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)
