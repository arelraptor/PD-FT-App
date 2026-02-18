
import VideoProcessingSingle as vp
import pickle
import numpy as np
import os
from tsfresh import extract_features

def get_evaluation(filename,video_id):

    data_final, frame_percentage = vp.processVideo(filename,video_id)

    print(frame_percentage)
    print("####")
    if frame_percentage > 90:
        #Aligning column order with the model's training schema
        order_columns_pkl_file = os.path.join("model", "order_colums.pkl")
        with open(order_columns_pkl_file, 'rb') as file:  
            order_columns = pickle.load(file)

        #Loading the feature dictionary as outputted by the tsfresh selection process.
        dict_pkl_file = os.path.join("model", "dictionary.pkl")
        with open(dict_pkl_file, 'rb') as file:  
            mydict = pickle.load(file)

        data_final["id"]="UploadedVideo"

        X = extract_features(data_final, kind_to_fc_parameters =mydict,column_id="id",n_jobs=0)

        model_pkl_file = os.path.join("model", "model.pkl")
        with open(model_pkl_file, 'rb') as file:  
            model = pickle.load(file)
        print("Model has been loaded")

        x_ordered=X[order_columns]

        value=model.predict(np.array(x_ordered))[0]
    else:
        value=-1

    return value

    
