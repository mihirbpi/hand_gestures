import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  while cap.isOpened():
    success, image = cap.read()

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            index_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            center_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            v_index = [index_landmark.x-center_landmark.x, index_landmark.y-center_landmark.y]
            v_middle = [middle_landmark.x-center_landmark.x, middle_landmark.y-center_landmark.y]
            mag_index = math.sqrt(v_index[0]**2 + v_index[1]**2)
            mag_middle = math.sqrt(v_middle[0]**2 + v_middle[1]**2)
            dot = v_index[0]*v_middle[0] + v_index[1]*v_middle[1]
            mags_prod = mag_index * mag_middle
            angle = math.acos(dot/mags_prod)

            if(0.2 <= angle <= 0.6 and v_index[1] <= center_landmark.y and v_middle[1] <= center_landmark.y and mag_index >= 0.33 and mag_middle >= 0.33):
                print(angle, "peace sign")

            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('MediaPipe Hands', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
