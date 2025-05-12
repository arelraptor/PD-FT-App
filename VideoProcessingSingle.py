import cv2
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import math

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

def processVideo(path_video,video_id):

    #definición de las columnas de coordenadas

    #columnas que marcarán las coordenadas del dedo índice
    cols_index=['INDEX_X','INDEX_Y','INDEX_Z']

    #columnas que marcarán las coordenadas del dedo pulgar
    cols_thumb=['THUMB_X','THUMB_Y','THUMB_Z']

    #columnas que marcarán las coordenadas de la muñeca
    cols_wrist=['WRIST_X','WRIST_Y','WRIST_Z']
    
  
    #inicializamos las variables finales
    data_final=pd.DataFrame()
    taps_done=pd.Series(dtype='float')

    bueno=0
    malo=0
    todos=0
    porcentaje=0
    image_saved=0
    
    base_options = BaseOptions(model_asset_path='model/hand_landmarker.task')
    options = HandLandmarkerOptions(base_options=base_options, running_mode=RunningMode.IMAGE, num_hands=1)
    detector = HandLandmarker.create_from_options(options)
    
    frames=[]
    frames_flip=[]
    manos=[]
    handedness=None
    
    cap = cv2.VideoCapture(path_video)
    
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using HandLandmarker
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)
        #print(results)
        #Draw the hand landmarks on the frame
        if results.handedness:
            manos.append(results.hand_landmarks)
            #print(results)
            hand_video=results.handedness[0][0].category_name
            bueno=bueno+1
            if image_saved==0 and (bueno+malo>20):
                annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), results)
                cv2.imwrite('app/static/screenshots/'+str(video_id)+'.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                image_saved = 1
            #bgr_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            #plt.imshow(cv2.cvtColor(bgr_frame,cv2.COLOR_RGB2BGR))
            #plt.show()
            #cv2.imshow("Frame", bgr_frame)
        else:
            malo=malo+1
    
    cap.release()
    
    todos=bueno+malo
    porcentaje=(bueno/(todos))*100

    df = pd.DataFrame()
    distancia_previa=0
    velocidad_previa=0
    for mano2 in manos:
        for hand_landmarks in mano2:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])

            temp_df= pd.DataFrame({'INDEX_X': [hand_landmarks_proto.landmark[8].x ],
                                   'INDEX_Y': [hand_landmarks_proto.landmark[8].y ],
                                   'INDEX_Z': [hand_landmarks_proto.landmark[8].z ],
                                   'THUMB_X': [hand_landmarks_proto.landmark[4].x ],
                                   'THUMB_Y': [hand_landmarks_proto.landmark[4].y ],
                                   'THUMB_Z': [hand_landmarks_proto.landmark[4].z ],
                                   'WRIST_X': [hand_landmarks_proto.landmark[0].x ],
                                   'WRIST_Y': [hand_landmarks_proto.landmark[0].y ],
                                   'WRIST_Z': [hand_landmarks_proto.landmark[0].z ]
                    })
            #calculamos el ángulo para cada combinación muñeca-índice vs muñeca-pulgar
            temp_df["angle"]=angle_of_vectors(temp_df.iloc[0][cols_wrist].values - temp_df.iloc[0][cols_index].values,temp_df.iloc[0][cols_wrist].values - temp_df.iloc[0][cols_thumb].values)
            #calculamos la distancia actual para calcular la velocidad
            distancia_actual=np.linalg.norm(temp_df.iloc[0][cols_index].values-temp_df.iloc[0][cols_thumb].values)
            velocidad_actual=abs(distancia_actual-distancia_previa)
            temp_df["velocidad"]=velocidad_actual
            distancia_previa=distancia_actual
            #calculamos la aceleración como la variación de la velocidad
            temp_df["aceleracion"]=abs(velocidad_actual-velocidad_previa)
            velocidad_previa=velocidad_actual
            df=pd.concat([df,temp_df], ignore_index=True)
        
    #Suavizamos coordenadas para el cálculo de la distancia
    df['INDEX_X']=savitzky_golay(np.array(df['INDEX_X']), 5, 3)
    df['INDEX_Y']=savitzky_golay(np.array(df['INDEX_Y']), 5, 3)
    df['INDEX_Z']=savitzky_golay(np.array(df['INDEX_Z']), 5, 3)
    df['THUMB_X']=savitzky_golay(np.array(df['THUMB_X']), 5, 3)
    df['THUMB_Y']=savitzky_golay(np.array(df['THUMB_Y']), 5, 3)
    df['THUMB_Z']=savitzky_golay(np.array(df['THUMB_Z']), 5, 3)
    df['WRIST_X']=savitzky_golay(np.array(df['WRIST_X']), 5, 3)
    df['WRIST_Y']=savitzky_golay(np.array(df['WRIST_Y']), 5, 3)
    df['WRIST_Z']=savitzky_golay(np.array(df['WRIST_Z']), 5, 3)
        
        
    #Calculamos la distancia entre el dedo índice y pulgar        
    df["Distancia"]=np.linalg.norm(df[cols_index].values - df[cols_thumb].values,axis=1)

        
    #Normalizamos la distancia entre 0 y 1
    max_value=df["Distancia"].max()
    min_value=df["Distancia"].min()
    df["Distancia_norm"]=(df["Distancia"] - min_value) / (max_value - min_value)
        
    df["Distancia_ang"]=df["angle"]/90
    #print(df)
        
    #Calculamos los picos y valles de la gráfica de distancia
    ilocs_min = argrelextrema(df.Distancia_norm.values, np.less_equal, order=8)[0]
    ilocs_max = argrelextrema(df.Distancia_norm.values, np.greater_equal, order=8)[0]

    #Calculamos los valle cuya distancia normalizada < 0.2
    y=df.iloc[ilocs_min].Distancia_norm < 0.2 

    #Acabamos los videos en el último valle
    df_cleaned=df[:y[y.values==True].index[-1]]

    #Empezamos todos los videos en el primer valle
    df_cleaned=df_cleaned[y[y.values==True].index[0]:]

    #Re-Calculamos los picos y valles de la gráfica de distancia
    ilocs_min = argrelextrema(df_cleaned.Distancia_norm.values, np.less_equal, order=8)[0]
    ilocs_max = argrelextrema(df_cleaned.Distancia_norm.values, np.greater_equal, order=8)[0]
    taps_single=(len(ilocs_max))
    #print(taps_single)
    taps_done=pd.concat([taps_done,pd.Series(taps_single)], ignore_index=True)    

    df_cleaned=df_cleaned.reset_index()
    temp_df_final=pd.DataFrame(df_cleaned[["Distancia_norm","Distancia_ang","angle","velocidad","aceleracion"]])
    

    data_final=pd.concat([data_final,temp_df_final], ignore_index=True)

    return data_final, porcentaje
    

#Función para caclular el ángulo recibidos dos vectores
def angle_of_vectors (vector_1,vector_2):

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return math.degrees(angle)


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError (msg):
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


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