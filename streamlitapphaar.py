import streamlit as st
from PIL import Image
import cv2
import numpy as np

def detect_faces(image):
    # Convert the image to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the Haar cascade XML file for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

    return image

st.title("Face Detector using Haar Cascade")

st.header("ABOUT")
st.write("Welcome, in this app you can upload any image and find number of faces present in the image. You will also be shown where the faces are present by marking them by a red box above the face.")

st.header("Steps to use the app:")
st.markdown("1) Open the sidebar")
st.markdown("2) Click on 'UPLOAD IMAGE' and choose image in which you want to find faces.")
st.markdown("3) The output will be shown on the main page.")


st.header("Upload Image")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded File", use_column_width=True)
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect faces in the image
        detected_image = detect_faces(img_cv)

        # Convert the image back to PIL format
        image = Image.fromarray(detected_image)

        # Display the detected faces
        st.image(image, caption="Detected Faces", use_column_width=True)


st.write("Made by Sameer")
