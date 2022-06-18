import cv2
import mediapipe as mp
import numpy as np
from math import sqrt
def cal_dist(p1,p2):
  return int(sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# For webcam input:
cap = cv2.VideoCapture(0)
success, image = cap.read()
image2 = np.zeros(image.shape, np.uint8)



with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image_height, image_width, _ = image.shape
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    # image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
     
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image0 = image.copy()


    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        index_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
        index_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height) 
        thumb_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)
        thumb_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)
        middle_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)
        middle_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)
        ring_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)
        ring_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

        if cal_dist((middle_x,middle_y),(thumb_x,thumb_y)) < 20:
          cv2.circle(image2,(index_x,index_y),2,(255,255,255),2)
        if cal_dist((index_x,index_y),(middle_x,middle_y)) < 20:
          cv2.circle(image2,(index_x,index_y),20,(0,0,0),-1)
        
        # print(cal_dist((ring_x,ring_y),(thumb_x,thumb_y))//2)
        # cv2.circle(image2,(image_width//2,image_height//2),cal_dist((image_width//2,image_height//2),(ring_x,ring_y)),(0,255,255),2)
          
    # Flip the image horizontally for a selfie-view display.
    final_image = np.concatenate((image2, image), axis=1)
    cv2.imshow('MediaPipe Hands', cv2.flip(final_image, 1))
    if cv2.waitKey(1) & 0xFF == 27:
      break
cap.release()