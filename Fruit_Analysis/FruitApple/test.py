import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import time
import urllib.request

# Load the saved model
model = tf.keras.models.load_model('Apple.keras')
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Adjust size to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class, predictions
# Specify the path to the image you want to predict
image_path = 'test.jpg'
className=['2 Fresh Apple','1 Fresh and 1 Rotton Apple','2 Rotton Apple']
cmdsName=['111','222','333']
# Make a prediction
predicted_class, predictions = predict_image(image_path)
print(predicted_class)
res="0a0"
# Show the prediction probabilities for each class
#print('Prediction probabilities:', predictions)
#print('Prediction :', className[predicted_class[0]])
#print('Prediction :', cmdsName[predicted_class[0]])
p=predictions[0]
a=predicted_class[0]
a=p[a]
print("a= "+str(a))
if a>0.5:
    fdata=cmdsName[predicted_class[0]]
    res="0a0"
    if '1' in fdata:
        res="2a0"
    elif '2' in fdata:
        res="1a1"
    else :
        res="0a2"
    img = image.load_img(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted Class: {className[predicted_class[0]]}')
    plt.show()
else:
    print("Not Detected")
page = urllib.request.urlopen('https://siddikue.bsite.net/PutData?p='+res)
print("Done")
print("Done")
print("Done")
# Display the image