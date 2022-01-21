import tensorflow_addons as tfa
import glob, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow.keras.layers as L
from vit_keras import vit, visualize, layers
from utils import *
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import sklearn


if __name__ == "__main__":

    seed_everything()

    dataset_path = "..\\..\\10-01-22_database_Aiello"
    model_name = "ViT"

    image_size = 224
    batch_size = 16
    epochs = 1
    visualize_map = False

    classes_list = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14"]

    X = []
    Y = []

    for c in classes_list:
        if os.path.isdir(os.path.join(dataset_path, c)):
            for file_name in glob.glob(os.path.join(dataset_path, c) + "//*.jpg"):
                image = cv2.imread(file_name, cv2.COLOR_BGR2RGB)
                if len(image.shape) < 3:
                    image = np.stack((image,) * 3, axis=-1)
                    print(image.shape)
                    print(file_name)

                image = cv2.resize(image, (image_size,  image_size))
                X.append(image)
                y = [0] * len(classes_list)
                y[classes_list.index(c)] = 1
                Y.append(y)

    X_full = np.asarray(X)
    X_full = vit.preprocess_inputs(X_full)
    y_full = np.asarray(Y)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_full, y_full, test_size=0.1, random_state=1)

    vit_model = vit.vit_l16(
            image_size=image_size,
            activation='softmax',
            pretrained=True,
            include_top=False,
            pretrained_top=False,
            classes=len(classes_list))

    vit_model.summary()

    model = tf.keras.Sequential([
            vit_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2048, activation=tf.nn.gelu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(classes_list), 'softmax')
                ],
        name='vision_transformer')

    model.summary()

    learning_rate = 1e-4

    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                     factor=0.2,
                                                     patience=4,
                                                     verbose=1,
                                                     min_delta=1e-4,
                                                     min_lr=1e-6,
                                                     mode='max')

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                     min_delta=1e-4,
                                                     patience=20,
                                                     mode='max',
                                                     restore_best_weights=True,
                                                     verbose=1)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="..\\Checkpoints\\ViT\\" + model_name + "-{epoch:02d}.hdf5",
                                                      monitor="val_accuracy",
                                                      verbose=1,
                                                      save_best_only=False,
                                                      save_weights_only=False,
                                                      mode="max",
                                                      period=10)

    callbacks = [earlystopping, checkpointer, reduce_lr]

    model.fit(x=X_train,
              y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              callbacks=callbacks)

    if visualize_map:

        i = 0
        for x, y in zip(X_test, y_test):
            label = classes_list[np.argmax(y)]
            # image_map = x * 255
            image_map = ((x + 1) * 127.5)
            image_map = np.uint8(image_map)

            prediction = model.predict(np.expand_dims(x, axis=0))
            label_pred = classes_list[np.argmax(prediction)]

            # image = cv2.imread(os.path.join(test_path, "A\\Pelvis00009_right_0.99.png"))
            # image = cv2.resize(image, (224, 224))
            # image *= 255
            attention_map = visualize.attention_map(model=model.get_layer("vit-l16"), image=image_map)

            # cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Original', 400, 400)
            # cv2.namedWindow('Attention', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Attention', 400, 400)
            # cv2.imshow('Original', image_map)
            # print("Original {}, Predicted {}".format(label, label_pred))
            # cv2.imshow("Attention", attention_map)
            # cv2.waitKey()

            cv2.imwrite("..\\Map\\Map-{}-{}-{}.png".format(label, label_pred, i), attention_map)
            i += 1

    print(model.evaluate(X_test, y_test, batch_size=batch_size))

    predicted_classes = np.argmax(model.predict(X_test), axis=1)
    true_classes = np.argmax(y_test, axis=1)

    confusionmatrix = confusion_matrix(true_classes, predicted_classes)
    print(confusionmatrix)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)
    plt.show()

    confusionmatrix_norm = np.around(confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis], decimals=2)
    print(confusionmatrix_norm)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix_norm, cmap='Blues', annot=True, cbar=True)
    plt.show()

    print(classification_report(true_classes, predicted_classes))

