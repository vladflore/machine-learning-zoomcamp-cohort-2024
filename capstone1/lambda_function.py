import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np
import json

preprocessor = create_preprocessor("xception", target_size=(299, 299))
interpreter = tflite.Interpreter(model_path="xception_08_0.977.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

classes = [
    "cane",
    "cavallo",
    "elefante",
    "farfalla",
    "gallina",
    "gatto",
    "mucca",
    "pecora",
    "ragno",
    "scoiattolo",
]
translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "dog": "cane",
    "elephant": "elefante",
    "butterfly": "farfalla",
    "chicken": "gallina",
    "cat": "gatto",
    "cow": "mucca",
    "spider": "ragno",
    "squirrel": "scoiattolo",
}


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_input(x):
    x /= 255.0
    # x -= 1.
    return x


def read_img(url):
    img = download_image(url)
    img = prepare_image(img, (299, 299))
    x = np.array(img, dtype="float32")
    X = np.array([x])
    X = preprocess_input(X)
    return X


def predict(url):
    X = preprocessor.from_url(url)
    # X = read_img(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return preds[0].tolist()


def lambda_handler(event, context):
    url = event["url"]
    preds = predict(url)
    all = dict(zip(classes, preds))
    predicted_animal = max(all, key=all.get)
    return json.dumps(
        {
            "animal": translate[predicted_animal]
        }
    )
