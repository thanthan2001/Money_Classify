import numpy as np
import streamlit as st
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Dinh nghia class
class_name = ['00000', '10000', '20000', '50000']

def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    # Dong bang cac layer
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Tao model
    input = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    # Them cac layer FC va Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='softmax', name='predictions')(x)

    # Compile
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

# Load weights model da train
my_model = get_model()
my_model.load_weights("weights-22-0.99.hdf5")

def preprocess_image(img):

    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_from_uploaded_file(uploaded_file):
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image for prediction
    img_array = preprocess_image(image_pil)

    # Dự đoán
    predict = my_model.predict(img_array)
    prediction_label = class_name[np.argmax(predict[0])]
    confidence = np.max(predict[0])

    st.write(f"Prediction: {prediction_label}")

def main():
    st.title("Money Classification Web App")
    st.write("Upload an image for prediction.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        predict_from_uploaded_file(uploaded_file)

if __name__ == "__main__":
    main()
