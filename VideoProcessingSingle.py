import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import math

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

def processVideo(path_video,video_id):

    #definición de las columnas de coordenadas

    #Columns index finger
    cols_index=['INDEX_X','INDEX_Y','INDEX_Z']
    #Columns thumb
    cols_thumb=['THUMB_X','THUMB_Y','THUMB_Z']
    #columnas wrist
    cols_wrist=['WRIST_X','WRIST_Y','WRIST_Z']
    
  
    #Variables for output files
    data_final=pd.DataFrame()

    good_frames_count=0
    bad_frames_count=0
    all_frames_count=0
    good_frames_percentage=0
    image_saved=0

    # Initialize MediaPipe options
    base_options = BaseOptions(model_asset_path='model/hand_landmarker.task')
    options = HandLandmarkerOptions(base_options=base_options, running_mode=RunningMode.IMAGE, num_hands=1)
    detector = HandLandmarker.create_from_options(options)

    frames = []
    hands = []
    handedness = None
    
    cap = cv2.VideoCapture(path_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        #frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using HandLandmarker
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        #Draw the hand landmarks on the frame
        if results.handedness:
            hands.append(results.hand_landmarks)
            hand_video=results.handedness[0][0].category_name
            good_frames_count= good_frames_count + 1

            if image_saved==0 and (good_frames_count + bad_frames_count > 20):
                annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), results)
                cv2.imwrite('app/static/screenshots/'+str(video_id)+'.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                image_saved = 1

        else:
            bad_frames_count= bad_frames_count + 1
    
    cap.release()
    
    all_frames_count = good_frames_count + bad_frames_count
    good_frames_percentage= (good_frames_count / (all_frames_count)) * 100

    if (good_frames_percentage > 90):
        df = pd.DataFrame()
        previous_distance = 0
        previous_speed = 0
        for hand in hands:
            for hand_landmarks in hand:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])

                temp_df = pd.DataFrame({'INDEX_X': [hand_landmarks_proto.landmark[8].x],
                                        'INDEX_Y': [hand_landmarks_proto.landmark[8].y],
                                        'INDEX_Z': [hand_landmarks_proto.landmark[8].z],
                                        'THUMB_X': [hand_landmarks_proto.landmark[4].x],
                                        'THUMB_Y': [hand_landmarks_proto.landmark[4].y],
                                        'THUMB_Z': [hand_landmarks_proto.landmark[4].z],
                                        'WRIST_X': [hand_landmarks_proto.landmark[0].x],
                                        'WRIST_Y': [hand_landmarks_proto.landmark[0].y],
                                        'WRIST_Z': [hand_landmarks_proto.landmark[0].z]
                                        })

                # Calculate the angle for each wrist-index vs. wrist-thumb combination
                temp_df["angle"] = angle_of_vectors(
                    temp_df.iloc[0][cols_wrist].values - temp_df.iloc[0][cols_index].values,
                    temp_df.iloc[0][cols_wrist].values - temp_df.iloc[0][cols_thumb].values)
                # Concat temp datagrame to the intermediate one
                df = pd.concat([df, temp_df], ignore_index=True)

        df["DISTANCE_ANG"] = df["angle"] / 90

        df['HAND_SIZE'] = df.apply(calculate_hand_size, axis=1)

        # Calculate the amplitude (distance between the thumb and the index finger)
        df['AMPLITUDE'] = df.apply(calculate_amplitude, axis=1)

        # Normalize the amplitude using the distance between the wrist and the index finger
        df['NORMALIZED_AMPLITUDE'] = (df['AMPLITUDE'] / df['HAND_SIZE'])

        # Smoothing of the normalized amplitude
        df['SMOOTHED_AMPLITUDE'] = df['NORMALIZED_AMPLITUDE'].rolling(window=3, center=True).mean()
        df['SMOOTHED_AMPLITUDE'] = df['SMOOTHED_AMPLITUDE'].fillna(method='bfill').fillna(method='ffill')

        # Define the frames per second (FPS) of the video
        delta_t = 1 / fps  # Time between frames in seconds

        # Velocity (first derivative)
        df['VELOCITY'] = df['SMOOTHED_AMPLITUDE'].diff() / delta_t
        df['VELOCITY'] = df['VELOCITY'].fillna(0)

        # Acceleration (second derivative)
        df['ACCELERATION'] = df['VELOCITY'].diff() / delta_t
        df['ACCELERATION'] = df['ACCELERATION'].fillna(0)

        peaks, _ = find_peaks(df['SMOOTHED_AMPLITUDE'], distance=fps // 3)  # minimum distance between peaks

        # Calculate the time of each peak
        peak_times = peaks * (1 / fps)  # Convert frame index to time in seconds
        # Calculate time differences between peaks
        periods = np.diff(peak_times)  # in seconds
        # Instantaneous frequency per pair of peaks (Hz)
        frequencies = 1 / periods

        # Adding to the DataFrame as a column (aligned with the second peak of each pair)
        df['FREQUENCY'] = np.nan
        df.loc[peaks[1:], 'FREQUENCY'] = frequencies

        temp_df_final = pd.DataFrame(df[["DISTANCE_ANG", "SMOOTHED_AMPLITUDE", "VELOCITY", "ACCELERATION", "FREQUENCY"]])

        data_final = pd.concat([data_final, temp_df_final], ignore_index=True)
        data_final['FREQUENCY'] = data_final['FREQUENCY'].fillna(0)

    return data_final, good_frames_percentage
    

def euclidean_distance(x1, y1, z1, x2, y2, z2):
    """
    Calculate the Euclidean distance between two 3D points.

    Args:
        x1, y1, z1 (float): Coordinates of the first point.
        x2, y2, z2 (float): Coordinates of the second point.

    Returns:
        float: Euclidean distance between the two points.
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

def angle_of_vectors (vector_1,vector_2):
    """
    Calculate the angle in degrees between two vectors in n-dimensional space.

    Args:
        vector_1 (np.array): First vector.
        vector_2 (np.array): Second vector.

    Returns:
        float: Angle between the two vectors in degrees.
    """
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return math.degrees(angle)


def calculate_hand_size(row):
    """
    Calculate the approximate size of the hand as the Euclidean distance between the wrist and the tip of the index finger.

    Args:
        row (pd.Series): A row from a DataFrame containing the following columns:
                         'WRIST_X', 'WRIST_Y', 'WRIST_Z', 'INDEX_X', 'INDEX_Y', 'INDEX_Z'.

    Returns:
        float: Euclidean distance between the wrist and the index finger tip.
    """
    wrist_index_distance = euclidean_distance(row['WRIST_X'], row['WRIST_Y'], row['WRIST_Z'],
                                              row['INDEX_X'], row['INDEX_Y'], row['INDEX_Z'])
    return wrist_index_distance


def calculate_amplitude(row):
    """
    Calculate the amplitude of finger movement as the Euclidean distance between the thumb tip and the index finger tip for a given frame.

    Args:
        row (pd.Series): A row from a DataFrame containing the following columns:
                         'THUMB_X', 'THUMB_Y', 'THUMB_Z',
                         'INDEX_X', 'INDEX_Y', 'INDEX_Z'.

    Returns:
        float: Euclidean distance between the thumb tip and the index finger tip.
    """
    thumb_index_distance = euclidean_distance(row['THUMB_X'], row['THUMB_Y'], row['THUMB_Z'],
                                              row['INDEX_X'], row['INDEX_Y'], row['INDEX_Z'])
    return thumb_index_distance

def draw_landmarks_on_image(rgb_image, detection_result):
    MARGIN = 10  # pixels
    FONT_SIZE = 3
    FONT_THICKNESS = 3
    HANDEDNESS_TEXT_COLOR = (255, 0, 0)  # RED

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image