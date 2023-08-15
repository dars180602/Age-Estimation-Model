# Loading the libraries
from keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
import io
import cv2

app = Flask(__name__)

# Loading the 2 models
model1 = load_model('age_model_acc_0.635.h5')
model2 = load_model('age_model_acc_0.587.h5')

# Dictionary mapping age group indices to their labels
age_groups = {
    0: "1-13",
    1: "13-18",
    2: "19-30",
    3: "31-42",
    4: "43-99"
}

# Function to perform face detection using OpenCV
def perform_face_detection(image):
    img_cv = np.array(image) # Convert the PIL image to a NumPy array for OpenCV processing
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY) # Convert the image to grayscale for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Load a pre-trained face detection model (Haarcascades)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # Detect faces in the image
    
    # Return appropriate messages based on face detection result
    if len(faces) == 0:
        return "No face detected!!!"
    elif len(faces) > 1:
        return "Multiple faces detected!!!"
    else:
        return None  # No issues with face detection

# Function to perform age prediction using trained models
def perform_age_prediction(image):
    img = image.resize((200, 200)) # Resize the input image to 200x200 pixels
    img = img.convert('L') # Convert the image to grayscale (Luminance) mode
    img_array = np.array(img) # Convert the image data to a NumPy array
    img_array = img_array.reshape((1, 200, 200, 1)) # Reshape the image array to match the input shape expected by the model (1 sample, 200x200 size, 1 channel)
    img_array = img_array / 255.0 # Normalize pixel values to the range [0, 1]
    
    # Perform age prediction using model1 and model2, and calculate weighted average of prediction probabilities
    age_prediction_probs_1 = model1.predict(img_array)[0]
    age_prediction_probs_2 = model2.predict(img_array)[0]
    age_prediction_probs = (0.635 * age_prediction_probs_1) + (0.587 * age_prediction_probs_2)
    
    # Find the index of the predicted age group with the highest probability
    predicted_age_group = np.argmax(age_prediction_probs)
    
    # Retrieve the age label corresponding to the predicted age group index
    predicted_age_label = age_groups[predicted_age_group]
    
    # Return the predicted age label
    return predicted_age_label


# Function to check permission based on predicted age label
def check_permission(predicted_age_label):
    # Check if the predicted age label falls in the age groups "1-13" or "13-18"
    if predicted_age_label == "1-13" or predicted_age_label == "13-18":
        return "Not Permitted to Enter!!!"  # If age is in specified groups, deny entry
    else:
        return "Permitted to Enter!!!"  # Otherwise, allow entry


# Route for the main index page
@app.route('/', methods=['GET', 'POST'])
def index():
    no_image_warning = None
    
    if request.method == 'POST':
        image = request.files.get('fileToUpload')
        if image:
            img = Image.open(image)
            age_prediction = perform_age_prediction(img)
            return render_template('photo.html', age_prediction=age_prediction)
        
        else:
            no_image_warning = "No image uploaded yet!!!"
            
    return render_template('index.html', no_image_warning=no_image_warning)

# Route for handling the image upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files.get('fileToUpload')
        if f:
            img = Image.open(io.BytesIO(f.read()))

            # Perform face detection
            face_detection_result = perform_face_detection(img)
            if face_detection_result:
                return render_template('index.html', face_detection_warning=face_detection_result)

            # Save the uploaded image to a temporary file
            img_path = 'static/uploads/uploaded_image.jpg'  # Change this to your desired path
            img.save(img_path)
            
            # Perform age prediction
            age_prediction = perform_age_prediction(img)
            permission = check_permission(age_prediction)

            return render_template('photo.html', uploaded_image=img_path, age_prediction=age_prediction, permission=permission)
        else:
            no_image_warning = "No image uploaded yet!!!"
            return render_template('index.html', no_image_warning=no_image_warning)
    else:
        return "Invalid request method."

# Run the app if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)