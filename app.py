from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained emotion recognition model (make sure to provide the correct path to your model file)
model = load_model('model\modelv1_new_1.h5')
print("Model loaded successfully")

# Emotion mapping (adjust according to your model's output classes)
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Neutral']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Get the base64 image data
    image_data = data['image']
    image_data = image_data.split(',')[1]  # Remove the 'data:image/jpeg;base64,' part

    # Decode the image
    image = Image.open(BytesIO(base64.b64decode(image_data)))

    # Convert image to numpy array
    image = np.array(image)

    # Preprocessing: Convert to grayscale if needed (depends on your model)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Assuming your model requires grayscale

    # Resize image to match model input size (for example, 48x48)
    image = cv2.resize(image, (48, 48))

    # Normalize the image
    image = image / 255.0

    # Add an extra dimension for channels (if model requires it)
    image = np.expand_dims(image, axis=-1)  # Single channel (grayscale)
    image = np.expand_dims(image, axis=0)   # Add batch dimension

    # Run prediction
    predictions = model.predict(image)
    
    # Debugging: Print the raw predictions
    print("Raw model output:", predictions)

    # Get the predicted emotion
    emotion_index = np.argmax(predictions)
    emotion = emotion_labels[emotion_index]

    # Return the emotion prediction
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
