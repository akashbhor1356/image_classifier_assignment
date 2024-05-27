from flask import Flask, request, render_template
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model, PCA object, and label encoder
svm = joblib.load('svm_model.pkl')
pca = joblib.load('pca_object.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Feature extraction functions
def extract_histogram_features_single(img):
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_edge_features_single(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = edges.flatten()
    return edges

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (128, 128))
            
            # Extract features
            hist_features = extract_histogram_features_single(img)
            edge_features = extract_edge_features_single(img)
            
            # Combine features
            features = np.hstack([hist_features, edge_features])
            
            # Apply PCA
            features_pca = pca.transform([features])
            
            # Predict
            prediction = svm.predict(features_pca)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            
            return f'The image is classified as: {predicted_label}'
    return '''
    <!doctype html>
    <title>Image Classification</title>
    <h1>Upload an Image for Classification</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)

