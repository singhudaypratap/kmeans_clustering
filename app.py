import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pickle
from flask import Flask, request, jsonify, render_template
import os

# Load the trained model
model_path = "kmeans_model.pkl"  # Ensure the model file is in the same directory
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file '{model_path}' not found! Ensure it's in the same directory.")

with open(model_path, 'rb') as f:
    kmeans = pickle.load(f)

# Initialize and fit StandardScaler on the Iris dataset
scaler = StandardScaler()
iris = load_iris()
scaler.fit(iris.data)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input dimensions
        if not all(k in data for k in ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']):
            return jsonify({'error': 'Missing required fields'}), 400

        new_data_point = np.array([
            data['sepalLength'],
            data['sepalWidth'],
            data['petalLength'],
            data['petalWidth']
        ]).reshape(1, -1)

        # Scale and predict
        scaled_new_data = scaler.transform(new_data_point)
        predicted_cluster = kmeans.predict(scaled_new_data)[0]

        # Map cluster number to cluster name
        cluster_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        predicted_cluster_name = cluster_names.get(predicted_cluster, "Unknown Cluster")

        return jsonify({'cluster': predicted_cluster_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Render requires the app to listen on PORT environment variable
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
