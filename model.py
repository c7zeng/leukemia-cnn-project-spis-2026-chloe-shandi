import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("saved_model.keras")


def crop_image(img):
    """Crops, resizes, and grayscales images."""
    
    crop_box = (75, 75, 375, 375)
    
    cropped_img = img.crop(crop_box)
    cropped_img = cropped_img.resize((128, 128))
    cropped_img = cropped_img.convert("L")
    
    return cropped_img


def test_model(file_path):

    input_im = Image.open(file_path)

    cropped_im = crop_image(input_im)

    test_pic = np.empty(
        (1, 128, 128, 1),
        dtype=np.uint8
    )

    for x_pixel in range(128):
        for y_pixel in range(128):
            test_pic[0, x_pixel, y_pixel, 0] = cropped_im.getpixel(
                (x_pixel, y_pixel)
            )

    single_test_prediction = model.predict(
        test_pic,
        verbose=0
    )

    pred = tf.argmax(
        single_test_prediction,
        axis=1
    ).numpy()[0]

    if pred == 0:
        result = "is likely healthy"
    else:
        result = "shows signs of acute leukemia"

    if "hem" in str(file_path).lower():
        ground_truth = "a healthy cell"
    elif "all" in str(file_path).lower():
        ground_truth = "a cell with leukemia"
    else:
        ground_truth = "unknown"

    confidence = single_test_prediction[0][pred] * 100

    return result, confidence, ground_truth