import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import glob, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from vit_keras import vit
from vit_keras import visualize
from utils import *
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K


if __name__ == "__main__":

    seed_everything()

    visualize_map = False
    test_overall = True

    shuffle = False
    if visualize_map:
        shuffle = True

    IMAGE_SIZE = 224
    BATCH_SIZE = 16

    test_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\'
    model_name = "D:\\7classes_l16-evenbiggerdense-oversampling-7classes-08" #"..\\Models\\ViT-supervised\\7classes_l16-evenbiggerdense"

    classes_list = ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"]

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                              validation_split=0)

    test_gen = datagen.flow_from_directory(directory=test_path,
                                           classes=classes_list,
                                           batch_size=BATCH_SIZE,
                                           seed=1,
                                           color_mode='rgb',
                                           shuffle=shuffle,
                                           class_mode='categorical',
                                           target_size=(IMAGE_SIZE, IMAGE_SIZE))

    vit_model = vit.vit_l16(
        image_size=IMAGE_SIZE,
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=len(classes_list))

    model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dense(1096, activation=tf.nn.gelu),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dense(len(classes_list), 'softmax')
        tf.keras.layers.Dense(4096, activation=tf.nn.gelu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(classes_list), 'softmax')
    ],
        name='vision_transformer')

    model.load_weights(model_name + ".hdf5")

    learning_rate = 1e-4

    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  metrics=['accuracy'])

    if visualize_map:
        x = test_gen.next()
        for i in range(test_gen.batch_size):
            image = x[0][i]
            label = x[1][i]
            label = classes_list[np.argmax(label)]
            image_map = image * 255
            image_map = np.uint8(image_map)

            prediction = model.predict(np.expand_dims(image, axis=0))
            label_pred = classes_list[np.argmax(prediction)]

            # image = cv2.imread(os.path.join(test_path, "A\\Pelvis00009_right_0.99.png"))
            # image = cv2.resize(image, (224, 224))
            # image *= 255
            attention_map = visualize.attention_map(model=model.get_layer("vit-l16"), image=image_map)

            # Plot results
            if label == "B3":
                cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Original', 400, 400)
                cv2.namedWindow('Attention', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Attention', 400, 400)
                cv2.imshow('Original', image_map)
                print("Original {}, Predicted {}".format(label, label_pred))
                cv2.imshow("Attention", attention_map)
                cv2.waitKey()

    if test_overall:

        print(model.evaluate_generator(test_gen))

        predicted_classes = np.argmax(model.predict_generator(test_gen, steps=test_gen.n // test_gen.batch_size + 1),
                                      axis=1)
        true_classes = test_gen.classes
        class_labels = list(test_gen.class_indices.keys())

        confusionmatrix = confusion_matrix(true_classes, predicted_classes)
        print(confusionmatrix)
        plt.figure(figsize=(16, 16))
        sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)
        plt.show()

        # confusionmatrix_norm = confusionmatrix / confusionmatrix.astype(np.float).sum(axis=1)
        # confusionmatrix_norm.round(decimals=2)
        confusionmatrix_norm = np.around(confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis], decimals=2)

        print(confusionmatrix_norm)
        plt.figure(figsize=(16, 16))
        sns.heatmap(confusionmatrix_norm, cmap='Blues', annot=True, cbar=True)
        plt.show()

        print(classification_report(true_classes, predicted_classes))