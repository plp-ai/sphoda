import cv2
import dlib
import numpy as np
import time

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
print("hello")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Function to calculate the Euclidean distance between two points
def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Track the previous frame's landmarks
prev_landmarks = None

# Initialize variables for attention tracking
attention_unit = 0
start_time = None
movement_threshold = 300
time_threshold = 15  # in seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        # Initialize landmarks_coords to an empty array to ensure it is always defined
        landmarks_coords = []

        for face in faces:
            landmarks = predictor(gray, face)
            
            # Extract landmark coordinates
            landmarks_coords = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            if prev_landmarks is not None:
                # Calculate the movement of landmarks
                movements = [euclidean_dist(landmarks_coords[i], prev_landmarks[i]) for i in range(len(landmarks_coords))]
                total_movement = sum(movements)
                print(f"Total movement: {total_movement}")

                # Check if the total movement is below the threshold
                if total_movement < movement_threshold:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= time_threshold:
                        attention_unit += 1
                        print(f"Attention Unit incremented: {attention_unit}")
                        start_time = None  # Reset start_time after incrementing
                else:
                    start_time = None  # Reset start_time if movement exceeds the threshold
            else:
                start_time = None  # Reset start_time if it's the first frame

            prev_landmarks = landmarks_coords
        
        # Display the frame with landmarks
        for (x, y) in landmarks_coords:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        cv2.imshow("Frame", frame)
        
        # Break the loop if any key is pressed
        if cv2.waitKey(1) != -1:
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print(f"Total Attention Units: {attention_unit}")
    cap.release()
    cv2.destroyAllWindows()
