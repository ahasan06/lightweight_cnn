from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the model
MODEL_PATH = "best_student_model.h5"
model = load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Ensure uploads folder exists
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)

    # Save the file
    try:
        file.save(filepath)
        print(f"File saved at: {filepath}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({'error': 'File could not be saved'}), 500

    # Preprocess the image
    try:
        img = image.load_img(filepath, target_size=(224, 224))  # Adjust based on your model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_confidence = np.max(predictions)

        # Define the prediction labels
        stage_labels = {
            0: "Begin",
            1: "Early",
            2: "Pre",
            3: "Pro"
        }

        # # Define a threshold for confidence (e.g., 95% for certainty)
        # CONFIDENCE_THRESHOLD = 0.6  # Confidence threshold for valid predictions
        # MAX_CONFIDENCE_THRESHOLD = 0.99  # Maximum confidence before flagging as uncertain

        # # If the confidence is too high (e.g., close to 100%), it's likely not a good prediction
        # if predicted_confidence >= MAX_CONFIDENCE_THRESHOLD:
        #     predicted_stage = "Unknown"  # Flag as unknown if the model is overconfident
        #     predicted_confidence = 0  # Reset confidence if overconfident
        # elif predicted_confidence < CONFIDENCE_THRESHOLD:
        #     predicted_stage = "Unknown"  # If confidence is low, classify as unknown
        # else:
        predicted_stage = stage_labels.get(int(predicted_class[0]), "Unknown Stage")

        # Clean up the file after processing
        os.remove(filepath)

        # Determine the message for the frontend
        if predicted_stage == "Unknown":
            message = "Leukemia not found"
        else:
            message = f"{predicted_stage}"

        return jsonify({
            'prediction': message,
            'accuracy': f"{predicted_confidence * 100:.2f}%"
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Error during prediction'}), 500


if __name__ == '__main__':
    app.run(debug=True)
