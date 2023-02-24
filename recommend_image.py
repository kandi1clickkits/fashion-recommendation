import os
import cv2
from PIL import Image
import numpy as np
import pickle
import tensorflow
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from flask_cors import cross_origin

import io
import base64

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


def recommend_image(uploaded_file):
    if os.path.isfile(uploaded_file):
        features = feature_extraction(os.path.join(uploaded_file), model)
        indices = recommend(features, feature_list)
        value = str()
        for i in indices[0]:
            img = Image.open(filenames[i])
            buffer = io.BytesIO()
            img.save(buffer, 'png')
            buffer.seek(0)
            data = buffer.read()
            data = base64.b64encode(data).decode()
            value += f'<br><img src="data:image/png;base64,{data}">' + "<br>"
        return value
    else:
        return "Some error occurred"
