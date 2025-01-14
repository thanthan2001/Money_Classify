import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from tkinter import filedialog
from tkinter import Tk

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

def predict_and_display(image_path):
    # Tiếp tục với quá trình predict
    image_org = cv2.imread(image_path)
    image_org = cv2.resize(image_org, dsize=(128, 128))
    image_org_display = image_org.copy()  # Tạo bản sao để hiển thị

    image_org = image_org.astype('float') * 1./255
    image = np.expand_dims(image_org, axis=0)

    # Tăng kích thước ảnh để xuất ra màn hình
    scale_percent = 200  # Tăng kích thước lên 200%
    width = int(image_org_display.shape[1] * scale_percent / 100)
    height = int(image_org_display.shape[0] * scale_percent / 100)
    dim = (width, height)
    image_org_display = cv2.resize(image_org_display, dim, interpolation=cv2.INTER_AREA)

    # Dự đoán
    predict = my_model.predict(image)
    prediction_label = class_name[np.argmax(predict[0])]

    confidence = np.max(predict[0])

    # Sử dụng định dạng chuỗi để hiển thị confidence với hai chữ số thập phân
    text = f"{prediction_label}"
    cv2.putText(image_org_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Hiển thị ảnh và giữ màn hình
    cv2.imshow("Predicted Image", image_org_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Sử dụng hộp thoại để chọn ảnh và hiển thị dự đoán
Tk().withdraw()
image_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])

if image_path:
    predict_and_display(image_path)
else:
    print("Không có ảnh được chọn.")
