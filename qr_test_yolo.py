import warnings

import numpy as np
import cv2 as cv
import pyttsx3
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dense, Dropout
K.clear_session()
classes = open('qrcode.names').read().strip().split('\n')
net = cv.dnn.readNetFromDarknet('qrcode-yolov3-tiny.cfg', 'qrcode-yolov3-tiny.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
ln = net.getLayerNames()
print(ln)
def QRcode(filename='2.jpg'):
    ## Defining batch specfications
    batch_size = 32
    img_height = 250
    img_width = 250
    base_dir_train = "./datasets"
    training_data = tf.keras.preprocessing.image_dataset_from_directory(base_dir_train,
                                                                        seed=42,
                                                                        image_size=(img_height, img_width),
                                                                        batch_size=batch_size
                                                                        )
    base_dir_test = "./datasets"

    validation_data = tf.keras.preprocessing.image_dataset_from_directory(base_dir_test,
                                                                          seed=42,
                                                                          image_size=(img_height, img_width),
                                                                          batch_size=batch_size
                                                                          )

    target_names = training_data.class_names
    print(target_names)
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(250, 250, 3),
                     kernel_size=(11, 11), strides=(4, 4),
                     padding='valid'))
    model.add(Activation('relu'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11, 11),
                     strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())
    # Flattening
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(250 * 250 * 3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    # Output Softmax Layer
    model.add(Dense(len(target_names)))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics = [keras.metrics.precision(), keras.metrics.recall()])

    epoch = 10
    history = model.fit(training_data, validation_data=validation_data, epochs=epoch)

    fig1 = plt.gcf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.axis(ymin=0.4, ymax=1)
    plt.grid()
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])
    plt.savefig('chart1.jpg')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid()
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])
    plt.savefig('chart2.jpg')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid()
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])
    plt.savefig('chart3.jpg')
    plt.show()

    import cv2
    image = cv2.imread(filename)

    image_resized = cv2.resize(image, (img_height, img_width))
    image = np.expand_dims(image_resized, axis=0)
    print(image.shape)
    pred = model.predict(image)
    print(pred)
    output_class = target_names[np.argmax(pred)]
    print("The predicted class is", output_class)
    return output_class

def text_to_speech(text, gender):
        """
        Function to convert text to speech
        :param text: text
        :param gender: gender
        :return: None
        """
        voice_dict = {'Male': 0, 'Female': 1}
        code = voice_dict[gender]

        engine = pyttsx3.init()

        # Setting up voice rate
        engine.setProperty('rate', 125)

        # Setting up volume level  between 0 and 1
        engine.setProperty('volume', 0.8)

        # Change voices: 0 for male and 1 for female
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[code].id)

        engine.say(text)
        engine.runAndWait()


from tifffile import askopenfilename

filename=askopenfilename()
print(filename)
res=QRcode(filename)
if res>0:
    img = cv.imread(filename)
    detector = cv.QRCodeDetector()
    data, bbox, straight_qrcode = detector.detectAndDecode(img)

    # if there is a QR code
    if bbox is not None:
        print(f"QRCode data:\n{data}")
        text_to_speech(data, 'Female')

