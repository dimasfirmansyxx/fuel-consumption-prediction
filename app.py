from flask import Flask, render_template,url_for,request,jsonify
from flask_cors import CORS, cross_origin
import os
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def getFuelConsumption(cylinders, displacement, horsepower, weight, model_year):
    values = [[cylinders,displacement,horsepower,weight,model_year]]
    scaler_path = os.path.join(os.path.dirname('models/'),'consumption-scaler.pkl')
    sc = None
    with open(scaler_path,'rb') as f:
        sc=pickle.load(f)
    values = sc.transform(values)
    model = load_model(r'models/consumption-model.h5')
    prediction = model.predict(values)
    prediction = float(prediction)
    return prediction

def getCarType(cylinders, displacement, horsepower, weight, model_year, kml):
    values = [[cylinders,displacement,horsepower,weight,model_year,kml]]
    with open(os.path.join(os.path.dirname('models/'),'type-model.pkl'), 'rb') as file :
        model = pickle.load(file)

    prediction = model.predict(values)
    prediction = int(prediction)

    if (prediction == 1):
        return 'MPV'
    elif (prediction == 2): 
        return 'SUV'
    elif (prediction == 3):
        return 'Hatchback'
    elif (prediction == 4):
        return 'Sedan'
    elif (prediction == 5):
        return 'Minivan'
    elif (prediction == 6):
        return 'Wagon'
    elif (prediction == 7):
        return 'Pickup'


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
@cross_origin()
def result():
    cylinders = int(request.form['cylinders'])
    displacement = int(request.form['displacement'])
    horsepower = int(request.form['horsepower'])
    weight = int(request.form['weight'])
    model_year = int(request.form['model_year'])

    fuel = getFuelConsumption(cylinders, displacement, horsepower, weight, model_year)
    carType = getCarType(cylinders, displacement, horsepower, weight, model_year, fuel)
    
    if (fuel > 20):
        score = 'Excellent'
    elif (fuel > 11):
        score = 'Good'
    else:
        score = 'Bad'

    json_dict={
        'consumption': fuel,
        'consumption_score': score,
        'type': carType
    }   
    
    return jsonify(json_dict)
    
if __name__=='__main__':
    app.run(debug=True,port=3298)