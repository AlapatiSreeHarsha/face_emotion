import streamlit as st
import cv2
import numpy as np
from fer import FER  # Import FER for emotion recognition

# Initialize the FER detector
detector = FER()

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
        
        # Use FER library to detect the emotion
        emotion, score = detector.top_emotion(face)
        
        # Draw a rectangle around the face and display emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if score is not None:
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
st.title("Live People Detection and Emotion Tracking")

# Add buttons to start and stop detection
start_button = st.button("Start Detection")
stop_button = st.button("Stop Detection")

# Initialize an empty container for displaying the webcam feed
frame_container = st.empty()

# Control detection state
if "run" not in st.session_state:
    st.session_state.run = False

if start_button:
    st.session_state.run = True

if stop_button:
    st.session_state.run = False

if st.session_state.run:
    # Open the webcam for live video
    cap = cv2.VideoCapture(0)

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.write("Error: Unable to access webcam.")
            break
        
        # Process the frame (detect faces and emotions)
        processed_frame = process_frame(frame)
        
        # Convert frame to RGB (Streamlit requires RGB format)
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Display the processed frame in Streamlit
        frame_container.image(frame_rgb, channels="RGB", use_column_width=True)
    
    cap.release()
else:
    st.write("Click 'Start Detection' to begin!")
