from flask import Flask, request, render_template, jsonify

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('PoductDemand.csv')

# Handle missing values
data['Total Price'].fillna(data['Total Price'].mean(), inplace=True)

# Split data into features and target variable
X = data[['Total Price', 'Base Price']]
y = data['Units Sold']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train DecisionTreeRegressor model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Train RandomForestRegressor model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the frontend form
        total_price = float(request.form['total_price'])
        base_price = float(request.form['base_price'])

        # Prepare features for prediction
        features = pd.DataFrame({'Total Price': [total_price], 'Base Price': [base_price]})
        
        # Make predictions using both models
        dt_prediction = model.predict(features)[0]
        rf_prediction = rf_model.predict(features)[0]
        
        # Return the predictions
        return jsonify({'decision_tree_prediction': dt_prediction, 'random_forest_prediction': rf_prediction})
    else:
        return jsonify({'error': 'Method Not Allowed'}), 405

if __name__ == '__main__':
    app.run(debug=True)
