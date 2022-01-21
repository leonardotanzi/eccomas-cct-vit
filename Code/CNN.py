import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation
import seaborn as sns
from utils import *


if __name__ == "__main__":

    seed_everything()

    model_name = "resnet"

    train_path = 'D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\Dataset\\FinalDataset\\Train\\'
    test_path = 'D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\Dataset\\FinalDataset\\Test\\'

    # train_path = "D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\Dataset\\Super\\"
    # test_path = "D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\Dataset\\Super\\"

    classes_list = ["YES", "NO"]

    IMAGE_SIZE = 224
    BATCH_SIZE = 8
    EPOCHS = 20
    load = False

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.15,
                                                              preprocessing_function=preprocess_input)

    train_gen = datagen.flow_from_directory(directory=train_path,
                                            subset="training",
                                            classes=classes_list,
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            color_mode='rgb',
                                            shuffle=True,
                                            class_mode='categorical',
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE))

    valid_gen = datagen.flow_from_directory(directory=train_path,
                                            subset='validation',
                                            classes=classes_list,
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            color_mode='rgb',
                                            shuffle=False,
                                            class_mode='categorical',
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE))

    test_gen = datagen.flow_from_directory(directory=test_path,
                                           classes=classes_list,
                                           batch_size=BATCH_SIZE,
                                           seed=1,
                                           color_mode='rgb',
                                           shuffle=False,
                                           class_mode='categorical',
                                           target_size=(IMAGE_SIZE, IMAGE_SIZE))

    learning_rate = 1e-4

    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

    model = InceptionV3(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights="imagenet", pooling="avg")
    out = Dense(4096, activation="relu")(model.output)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = Dense(len(classes_list))(out)
    out = Activation("softmax")(out)

    model = Model(inputs=model.input,
                  outputs=out)

    model.summary()

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
    STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                     factor=0.2,
                                                     patience=2,
                                                     verbose=1,
                                                     min_delta=1e-4,
                                                     min_lr=1e-6,
                                                     mode='max')

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                     min_delta=1e-4,
                                                     patience=10,
                                                     mode='max',
                                                     restore_best_weights=True,
                                                     verbose=1)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="D:\\" + model_name + "-{epoch:02d}.hdf5",
                                                      monitor="val_accuracy",
                                                      verbose=1,
                                                      save_best_only=False,
                                                      save_weights_only=False,
                                                      mode="max",
                                                      period=10)

    callbacks = [reduce_lr, checkpointer]

    if load:
        model.load_weights(model_name + ".hdf5")
    else:
        model.fit(x=train_gen,
                  steps_per_epoch=STEP_SIZE_TRAIN,
                  validation_data=valid_gen,
                  validation_steps=STEP_SIZE_VALID,
                  epochs=EPOCHS,
                  callbacks=callbacks)

        model.save(model_name + ".hdf5")

    print(model.evaluate_generator(test_gen))

    predicted_classes = np.argmax(model.predict_generator(test_gen, steps=test_gen.n // test_gen.batch_size + 1),
                                  axis=1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    confusionmatrix = confusion_matrix(true_classes, predicted_classes)
    print(confusionmatrix)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)

    confusionmatrix_norm = np.around(confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis],
                                     decimals=2)
    print(confusionmatrix_norm)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix_norm, cmap='Blues', annot=True, cbar=True)
    plt.show()

    print(classification_report(true_classes, predicted_classes))