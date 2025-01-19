import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace  # Using DeepFace for emotion recognition

# Load the face cascade for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_and_emotions(frame):
    """Detect faces and classify emotions for each detected face."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Count the number of faces detected
    num_people = len(faces)
    
    for (x, y, w, h) in faces:
        # Extract face region
        face = frame[y:y+h, x:x+w]
        
        # Use DeepFace to detect the emotion
        result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        score = result[0]['emotion'][emotion]
        
        # Draw a rectangle around the face and display emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion} ({score*100:.1f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return frame, num_people

def process_frame(frame):
    """Process each frame to detect faces and emotions."""
    # Detect faces and emotions
    frame, num_people = detect_faces_and_emotions(frame)
    
    # Add text to display the number of people
    cv2.putText(frame, f"People Count: {num_people}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

# Streamlit Web App UI
st.title("People Detection and Emotion Tracking")

# Upload button for image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read the uploaded image
    img = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
    
    # Process the frame (detect faces and emotions)
    processed_frame = process_frame(frame)
    
    # Convert frame to RGB (Streamlit requires RGB format)
    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    # Display the processed frame in Streamlit
    st.image(frame_rgb, channels="RGB", use_column_width=True)
else:
    st.write("Upload an image to begin detection.")
