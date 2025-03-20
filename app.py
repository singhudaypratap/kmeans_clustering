# prompt: import pickle
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import load_iris
# from flask import Flask, request, jsonify, render_template
# from pyngrok import ngrok  # Using pyngrok for public URL
# import os
# # Initialize Flask app
# app = Flask(__name__, template_folder='/content/templates')  # Ensure templates are in this folder
# # Load the trained model
# model_path = "/content/kmeans_model.pkl"  # Ensure the file is uploaded in Colab
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"❌ Model file '{model_path}' not found! Upload it to Colab.")
# with open(model_path, 'rb') as f:
#     kmeans = pickle.load(f)
# # Initialize and fit StandardScaler on Iris dataset
# scaler = StandardScaler()
# iris = load_iris()
# scaler.fit(iris.data)
# @app.route('/')
# def index():
#     return render_template('index.html')  # Serve the HTML page
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     new_data_point = np.array(data['data']).reshape(1, -1)  # Reshape to 2D array
#     scaled_new_data = scaler.transform(new_data_point)
#     predicted_cluster = kmeans.predict(scaled_new_data)[0]
#     # Map cluster number to cluster name
#     cluster_names = {
#         0: 'Setosa',
#         1: 'Versicolor',
#         2: 'Virginica'
#     }
#     predicted_cluster_name = cluster_names[predicted_cluster]
#     return jsonify({'cluster': predicted_cluster_name})
# if __name__ == '__main__':
#     port = 5000
#     ngrok.kill()
#     public_url = ngrok.connect(port)
#     print(f"🌍 Public URL: {public_url}")  # Print the public URL for access
#     app.run(port=port)
# convert this to app.py for deployment using render and github
# convert this to app.py for deployment using render and github

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pickle
from flask import Flask, request, jsonify, render_template
import os

# Load the trained model
model_path = "kmeans_model.pkl"  # Ensure the file is in the same directory
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file '{model_path}' not found! Ensure it's in the same directory.")

with open(model_path, 'rb') as f:
    kmeans = pickle.load(f)

# Initialize and fit StandardScaler on Iris dataset
scaler = StandardScaler()
iris = load_iris()
scaler.fit(iris.data)

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    new_data_point = np.array(data['data']).reshape(1, -1)
    scaled_new_data = scaler.transform(new_data_point)
    predicted_cluster = kmeans.predict(scaled_new_data)[0]

    # Map cluster number to cluster name
    cluster_names = {
        0: 'Setosa',
        1: 'Versicolor',
        2: 'Virginica'
    }
    predicted_cluster_name = cluster_names[predicted_cluster]

    return jsonify({'cluster': predicted_cluster_name})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
