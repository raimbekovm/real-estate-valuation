# External libraries
import os
import datetime
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from geopy.distance import geodesic
import pickle

# Statistical analysis and modeling
from catboost import CatBoostRegressor


# Importing necessary functions from data_mining module
from functions import geocode_2gis, count_places_within_radius, checking_park


# Committing RANDOM_SEED to make experiments repeatable
SEED = 42


# Create flask app
flask_app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

# Function to process data and to create new features 
def process_data(data):
    columns = [
    'owner', 'complex_name', 'house_type', 'in_pledge', 'construction_year',
    'ceiling_height', 'bathroom_info', 'condition', 'area', 'room_count',
    'floor', 'floor_count', 'district', 'complex_class', 'parking',
    'elevator', 'schools_within_500m', 'kindergartens_within_500m',
    'park_within_1km', 'distance_to_center', 'distance_to_botanical_garden',
    'distance_to_triathlon_park', 'distance_to_astana_park',
    'distance_to_treatment_facility', 'distance_to_railway_station_1',
    'distance_to_railway_station_2', 'distance_to_industrial_zone',
    'last_floor', 'first_floor'
    ]
    
    owner = data["owner"]        
    complex_name = data["complex_name"]
    house_type = data["house_type"]
    in_pledge = data["in_pledge"] == 'yes'
    construction_year = int(data["construction_year"])
    ceiling_height = float(data["ceiling_height"])
    ceiling_height = min([2.5, 2.7, 2.8, 2.9, 3, 3.3, 3.5, 4], key=lambda x: abs(x - ceiling_height))
    bathroom_info = data["bathroom_info"]
    condition = data["condition"]
    area = float(data["area"])
    room_count = int(data["room_count"])
    floor = int(data["floor"])
    floor_count = int(data["floor_count"])  
    district = data["district"]
    complex_class = data["complex_class"]
    parking = data["parking"]
    elevator = data["elevator"]
    
    address = data["address"]
    coordinates_str = geocode_2gis(address)
    coordinates_list = coordinates_str.replace('(', '').replace(')', '') 
    latitude = float(coordinates_list[0])
    longitude = float(coordinates_list[1])
    coordinates = (latitude, longitude)
    
    schools_within_500m = float(count_places_within_radius("school", coordinates))
    schools_within_500m = min(schools_within_500m, 4)
    
    kindergartens_within_500m = float(count_places_within_radius("kindergarten", coordinates))
    kindergartens_within_500m = min(kindergartens_within_500m, 3)
    
    park_within_1km = checking_park(coordinates)
    geo_center_of_astana = (51.128318, 71.430381)
    distance_to_center = geodesic(geo_center_of_astana, coordinates).kilometers
    botanical_garden = (51.106433, 71.416329)
    distance_to_botanical_garden = geodesic(botanical_garden, coordinates).kilometers
    triathlon_park = (51.13593, 71.449809)
    distance_to_triathlon_park =  geodesic(triathlon_park, coordinates).kilometers
    astana_park = (51.156264, 71.419961)
    distance_to_astana_park = geodesic(astana_park, coordinates).kilometers
    treatment_facility = (51.144302, 71.337247)
    distance_to_treatment_facility = geodesic(treatment_facility, coordinates).kilometers
    railway_station_1 = (51.195572, 71.409173)
    distance_to_railway_station_1 = geodesic(railway_station_1, coordinates).kilometers
    railway_station_2 = (51.112488, 71.531596)
    distance_to_railway_station_2 = geodesic(railway_station_2, coordinates).kilometers
    industrial_zone = (51.140231, 71.551219)
    distance_to_industrial_zone = geodesic(industrial_zone, coordinates).kilometers
    
    last_floor = floor == floor_count
    first_floor = floor == 1 or (floor == 2 and parking == 'underground')

    data_for_df = [
        owner, complex_name, house_type, in_pledge, construction_year, ceiling_height, bathroom_info, 
        condition, area, room_count, floor, floor_count, district, complex_class, parking, elevator, 
        schools_within_500m, kindergartens_within_500m, park_within_1km, distance_to_center, 
        distance_to_botanical_garden, distance_to_triathlon_park, distance_to_astana_park, 
        distance_to_treatment_facility, distance_to_railway_station_1, distance_to_railway_station_2, 
        distance_to_industrial_zone, last_floor, first_floor
    ]
    
    processed_df = pd.DataFrame([data_for_df], columns=columns)
    return processed_df


@flask_app.route('/data/<path:path>')
def send_static(path):
    return send_from_directory('data', path)
        
@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()
    processed_data = process_data(form_data)
    prediction = model.predict(processed_data)
    prediction = np.exp(prediction)
    prediction_text = "Примерная стоимость: {:,.0f} тенге".format(prediction[0])
    return redirect(url_for('prediction_result', prediction_text=prediction_text))

@flask_app.route("/prediction_result")
def prediction_result():
    prediction_text = request.args.get('prediction_text', '')
    return render_template("prediction_result.html", prediction_text=prediction_text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    flask_app.run(debug=True,host='0.0.0.0',port=port)
    
#gunicorn -w 4 -b 0.0.0.0:8080 run_gunicorn:flask_app 