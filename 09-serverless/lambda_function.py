import tflite_runtime.interpreter as tflite
#from keras_image_helper import create_preprocessor

from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

#preprocessor = create_preprocessor('xception', target_size=(200, 200))
interpreter = tflite.Interpreter(model_path='model_2024_hairstyle_v2.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    x /= 255.0
    # x -= 1.
    return x

def read_img(url):
    img=download_image(url)
    img=prepare_image(img,(200,200))
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = preprocess_input(X)
    return X


def predict(url):
    #X = preprocessor.from_url(url)
    X = read_img(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return float(preds[0][0])

def lambda_handler(event, context):
    url = event['url']
    return predict(url)
