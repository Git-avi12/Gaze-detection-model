import dlib
import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import deque
import time
import tensorflow as tf

# Constants for blink detection and gaze smoothing
EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES = 3
LOOK_DOWN_THRESHOLD = 0.6  # Threshold for looking down (60% of eye height)
GAZE_THRESHOLD = 20  # Minimum pixel movement to register a new gaze direction
BUFFER_SIZE = 5  # Number of frames to track for smoothing
STARE_THRESHOLD = 3  # Time in seconds to detect staring at the same position
MARGIN_ERROR = 10  # Pixels margin of error for pupil position fluctuations

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/CAREER FOCUSED COURSES/Hysteresis Internship/shape_predictor_68_face_landmarks.dat')

# Load TensorFlow model for object detection
PATH_TO_CKPT = 'ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'  # Adjust path to your model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Start webcam capture
cap = cv2.VideoCapture(0)

# Blink tracking variables
blink_counter = 0
blink_frame_counter = 0
flag = 1  # Controls text display
gaze_buffer_left = deque(maxlen=BUFFER_SIZE)
gaze_buffer_right = deque(maxlen=BUFFER_SIZE)

# Variables for stare detection
stare_timer = 0
staring = False
previous_left_pupil = None
previous_right_pupil = None

# Function to detect pupil using contour method
def detect_pupil(eye_roi):
    if eye_roi.size == 0:
        return None  

    eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    eye_gray = cv2.GaussianBlur(eye_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(eye_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return cX, cY
    return None

# Function to get combined gaze direction based on pupil position
def get_combined_gaze_direction(left_pupil_x, left_pupil_y, left_eye_w, left_eye_h, right_pupil_x, right_pupil_y, right_eye_w, right_eye_h):
    horizontal_gaze = "Center"
    vertical_gaze = "Center"

    # Left eye gaze
    if left_pupil_x < left_eye_w * 0.35:  # Adjusted to register "Left" more easily
        horizontal_gaze = "left"
    elif left_pupil_x > left_eye_w * 0.65:  # Adjusted to register "Right" more easily
        horizontal_gaze = "right"
    
    if left_pupil_y < left_eye_h * 0.25:  # Reduced threshold for "Up"
        vertical_gaze = "Up"
    elif left_pupil_y > left_eye_h * 0.6:
        vertical_gaze = "Down"

    # Right eye gaze (this should be swapped for correct left/right detection)
    if right_pupil_x < right_eye_w * 0.35:  # Adjusted to register "Left" more easily
        horizontal_gaze = "right"  # This should be "right" because it's for the right eye
    elif right_pupil_x > right_eye_w * 0.65:  # Adjusted to register "Right" more easily
        horizontal_gaze = "left"  # This should be "left" because it's for the right eye
    
    if right_pupil_y < right_eye_h * 0.25:  # Reduced threshold for "Up"
        vertical_gaze = "Up"
    elif right_pupil_y > right_eye_h * 0.6:
        vertical_gaze = "Down"

    # Determine the final combined gaze
    if horizontal_gaze == "left" or horizontal_gaze == "right":
        return horizontal_gaze
    elif vertical_gaze == "Up" or vertical_gaze == "Down":
        return vertical_gaze
    else:
        return "Center"

# Function to smooth the gaze direction
def smooth_gaze_direction(new_gaze, gaze_buffer):
    if not gaze_buffer:
        gaze_buffer.append(new_gaze)
        return new_gaze

    most_common_gaze = max(set(gaze_buffer), key=gaze_buffer.count)

    if new_gaze != most_common_gaze:
        gaze_buffer.append(new_gaze)

    return most_common_gaze

# Function to check if the user is staring at the same spot
def check_staring(current_left_pupil, current_right_pupil):
    global previous_left_pupil, previous_right_pupil, stare_timer, staring

    if previous_left_pupil and previous_right_pupil:
        # Calculate the distance between the previous and current pupil positions
        left_distance = dist.euclidean(previous_left_pupil, current_left_pupil)
        right_distance = dist.euclidean(previous_right_pupil, current_right_pupil)

        # If the pupils haven't moved significantly (within the margin error), continue the staring detection
        if left_distance < MARGIN_ERROR and right_distance < MARGIN_ERROR:
            stare_timer += 1
        else:
            stare_timer = 0  # Reset timer if there's a significant movement
            staring = False
    else:
        stare_timer = 0  # Reset on first frame

    # If the timer exceeds the threshold, trigger the staring alert
    if stare_timer > STARE_THRESHOLD * 30:  # Assuming 30 FPS
        staring = True
    previous_left_pupil = current_left_pupil
    previous_right_pupil = current_right_pupil

    return staring

# Function to check for head tilt based on yaw and roll
def check_head_tilt(yaw, roll):
    if (yaw < -85 and yaw > -90) and (roll < 100 and roll > 80):
        return True
    return False

# Run the object detection on frame
def detect_objects(frame, detection_graph):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Prepare the frame
            image_np = np.expand_dims(frame, axis=0)
            image_np = np.asarray(image_np, dtype=np.uint8)

            # Get tensors for detection
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Run the object detection
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np}
            )

            return boxes, scores, classes, num_detections

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])

        left_eye_x, left_eye_y, left_eye_w, left_eye_h = cv2.boundingRect(left_eye_points)
        right_eye_x, right_eye_y, right_eye_w, right_eye_h = cv2.boundingRect(right_eye_points)

        left_eye_roi = frame[left_eye_y:left_eye_y + left_eye_h, left_eye_x:left_eye_x + left_eye_w]
        right_eye_roi = frame[right_eye_y:right_eye_y + right_eye_h, right_eye_x:right_eye_x + right_eye_w]

        left_pupil = detect_pupil(left_eye_roi)
        right_pupil = detect_pupil(right_eye_roi)

        # Object detection for electronic devices (phone, laptop, etc.)
        boxes, scores, classes, num_detections = detect_objects(frame, detection_graph)

        # Draw bounding boxes around detected objects
        for i in range(int(num_detections[0])):
            if scores[0][i] > 0.5:  # If detection confidence is high
                class_id = int(classes[0][i])
                box = boxes[0][i]

                # Filter by device-related class IDs (you can use the COCO class IDs for phones, laptops, etc.)
                if class_id in [67, 72, 74]:  # Example class IDs for devices like laptops, phones, etc. from the COCO dataset
                    (ymin, xmin, ymax, xmax) = box
                    (xmin, xmax, ymin, ymax) = (int(xmin * frame.shape[1]), int(xmax * frame.shape[1]), int(ymin * frame.shape[0]), int(ymax * frame.shape[0]))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f"Device Detected", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Rest of your blink and gaze detection code

    # Show the frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
