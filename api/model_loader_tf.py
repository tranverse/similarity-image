# model_loader_tf.py (TensorFlow)
from tensorflow.keras.models import load_model

def load_vgg16_keras_model(keras_path):
    model = load_model(keras_path, compile=False)
    return model
