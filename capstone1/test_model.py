import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input

# Use the model obtained as a result of running train.py
MODEL_NAME = "xception_08_0.977.keras"
trained_model = keras.models.load_model(MODEL_NAME)

# test_image = "butterfly.jpeg"
# test_image = "cat.jpeg"
test_image = "dog.jpeg"

input_size = 299

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
img = load_img(test_image, target_size=(input_size, input_size))
x = np.array(img)
X = np.array([x])
X = preprocess_input(X)
pred = trained_model.predict(X)
all = dict(zip(classes, pred[0]))
predicted_animal = max(all, key=all.get)
print(f"Predicted animal: {translate[predicted_animal]}")
