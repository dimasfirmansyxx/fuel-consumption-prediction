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
    
    values = [[cylinders,displacement,horsepower,weight,model_year]]

    scaler_path = os.path.join(os.path.dirname('models/'),'scaler.pkl')

    sc = None
    with open(scaler_path,'rb') as f:
        sc=pickle.load(f)
        
    values = sc.transform(values)

    model = load_model(r'models/model.h5')

    prediction = model.predict(values)
    prediction = float(prediction)
    print(prediction)
    
    json_dict={
        'prediction':prediction
    }   
    
    return jsonify(json_dict)
    
if __name__=='__main__':
    app.run(debug=True,port=3298)