import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)
print(keras.__version__)

ROOT = "content/drive/mydrive/mlzoomcamp/capstone2"
train_dir = f"{ROOT}/dataset/train/materials"
val_dir = f"{ROOT}/dataset/val/materials"


def make_model(input_size=150, learning_rate=0.01, size_inner=100, droprate=0.5):
    base_model = Xception(
    weights='imagenet', include_top=False, input_shape=(input_size, input_size, 3)
    )
    base_model.trainable = False
    ###########################################
    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    outputs = keras.layers.Dense(9)(drop)
    ###########################################
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss = loss, metrics=['accuracy'])
    return model

input_size = 299

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=10,
    zoom_range=0.1,
    vertical_flip=False,
    horizontal_flip=True,
)
train_ds = train_gen.flow_from_directory(
    train_dir, target_size=(input_size, input_size), batch_size=32
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_ds = val_gen.flow_from_directory(
    val_dir, target_size=(input_size, input_size), batch_size=32, shuffle=False
)

checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_{epoch:02d}_{val_accuracy:.3f}.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

learning_rate = 0.001
size_inner = 1000
droprate = 0.2

model = make_model(input_size=input_size, learning_rate=learning_rate, size_inner=size_inner, droprate=droprate)
model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=[checkpoint])
