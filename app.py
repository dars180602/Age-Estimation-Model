from keras.models import load_model
from flask import Flask, render_template, request, Response
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('age_model_acc_0.635.h5')  # Load your age estimation model

def perform_age_prediction(image):
    img = image.resize((200, 200))
    img = img.convert('L')
    img_array = np.array(img)
    img_array = img_array.reshape((1, 200, 200, 1))
    img_array = img_array / 255.0
    age_prediction = model.predict(img_array)
    age_prediction = int(age_prediction[0][0])
    return age_prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    no_image_warning = None  # Initialize the warning message variable
    
    if request.method == 'POST':
        image = request.files.get('fileToUpload')  # Use get method to prevent KeyError
        
    return render_template('index.html', no_image_warning=no_image_warning)


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files.get('fileToUpload')  # Use get method to prevent KeyError
        if f:
            img = Image.open(io.BytesIO(f.read()))

            # Save the uploaded image to a temporary file
            img_path = 'static/uploads/uploaded_image.jpg'  # Change this to your desired path
            img.save(img_path)
            
            # Perform age prediction
            age_prediction = perform_age_prediction(img)

            return render_template('photo.html', uploaded_image=img_path, age_prediction=age_prediction)
        else:
            no_image_warning = "No image uploaded yet!!!"
            return render_template('index.html', no_image_warning=no_image_warning)
    else:
        return "Invalid request method."


if __name__ == '__main__':
    app.run(debug=True)
