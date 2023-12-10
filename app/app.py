from flask import Flask, render_template,jsonify, request
import joblib
import numpy as np
import pandas as pd

# app.py
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_country', methods=['POST'])
def select_country():
    country = request.form.get('country')
    
    # Add logic to customize the available cities based on the selected country
    # For simplicity, let's assume India has the same three cities for now
    if country == 'India':
        return redirect(url_for('page2'))
    else:
        # Handle other countries or provide an error message
        return render_template('index.html', error='Invalid country selected')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the scaler and model based on the selected city
    city = request.form.get('city')

    # Adjust the paths based on your project structure
    scaler_x = joblib.load(f'scaler_x_{city.lower()}.joblib')
    scaler_y = joblib.load(f'scaler_y_{city.lower()}.joblib')  # Optional: Load scaler_y
    model = joblib.load(f'model_{city.lower()}.joblib')

    # Get user input
    area = float(request.form.get('area'))
    bhk = int(request.form.get('bhk'))
    bathroom = int(request.form.get('bathroom'))
    parking = int(request.form.get('parking'))
    furnishing = request.form.get('furnishing')

    # Convert furnishing to numerical representation
    furnishing_mapping = {'Furnished': 3, 'Semi Furnished': 1, 'Unfurnished': 2}
    furnishing_numeric = furnishing_mapping.get(furnishing, 0)

    # Preprocess input using the loaded scalers
    # new_X = pd.DataFrame({'Area': area, 'BHK': bhk,'Bathroom': bathroom,'Parking': parking,'Furnishing': furnishing_numeric})

    # # Standardize the new features using the same scaler from training
    # new_X_scaled = scaler_x.transform(new_X)

    # # Make predictions on the new feature vector
    # new_y_pred_scaled = model.predict(new_X_scaled)
    # new_y_pred = scaler_y.inverse_transform(new_y_pred_scaled.reshape(-1, 1)).ravel()
    
    features = scaler_x.transform([[area, bhk, bathroom, parking, furnishing_numeric]])

    # Make prediction
    prediction = model.predict(features)

    # Optional: If you need to inverse transform the prediction (if using scaler_y)
    if 'scaler_y' in locals():
        prediction = scaler_y.inverse_transform(prediction.reshape(-1, 1))

    result_dict = {
        'City': city,
        'Area': area,
        'BHK': bhk,
        'Bathroom': bathroom,
        'Parking': parking,
        'Furnishing': furnishing,
        'Predicted Price': prediction[0][0].astype(int)  # Assuming prediction is a 2D array
    }

    result_dict = {key: int(value) if isinstance(value, np.int32) else value for key, value in result_dict.items()}

    return jsonify(result_dict)

    # return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

