import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from pathlib import Path
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras.models import load_model

model = load_model("saved_model.keras")

# crop image function

def crop_image(img):
    '''crops, resizes, and grayscales images'''
    crop_box = (75, 75, 375, 375)
    cropped_img = img.crop(crop_box)
    cropped_img = cropped_img.resize((128,128)) #resize
    cropped_img = cropped_img.convert('L') #returns a black and white image
    return cropped_img

def test_model(file_path):
    input_im = Image.open(file_path)
    cropped_im = crop_image(input_im)

    test_pic = np.empty((1,128,128,1), dtype=np.uint8)
    for x_pixel in range(128):
        for y_pixel in range(128):
            
            test_pic[0,x_pixel, y_pixel, 0] = cropped_im.getpixel((x_pixel, y_pixel))
            
    single_test_prediction = model.predict(test_pic)

    pred = tf.argmax(single_test_prediction, axis=1).numpy()[0]
    if pred == 0:
        result = "is likely healthy"
    else:
        result = "shows signs of acute leukemia"


    if "hem" in str(file_path):
        ground_truth = "a healthy cell"

    elif "all" in str(file_path):
        ground_truth = "a cell with leukemia"
    

    print("the model predicts this cell", result)
    print("confidence:", (max(single_test_prediction[0][0], (1-single_test_prediction[0][0])))*100,"%")
    print("an oncologist has identified this cell as", ground_truth)
