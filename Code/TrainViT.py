import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import glob, random, os, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from utils import *
import sklearn
import tensorflow as tf
from tensorflow import keras
from vit_keras import vit, layers as layersvit


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        #
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])

        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def vision_transformer():
    inputs = layers.Input(shape=(image_size, image_size, 3))

    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)  #num_patches
    encoded_patches = layersvit.ClassToken(name="class_token")(encoded_patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    # Classify outputs.
    logits = layers.Dense(n_classes)(features)

    # Create the model.
    model = tf.keras.Model(inputs=inputs, outputs=logits)

    return model


def show_patches(X):

    plt.figure(figsize=(4, 4))
    image = X[0]
    plt.imshow(image.astype('uint8'))
    plt.axis('off')
    plt.show()

    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(image_size, image_size)
    )
    patches = Patches(patch_size)(resized_image)
    print(f'Image size: {image_size} X {image_size}')
    print(f'Patch size: {patch_size} X {patch_size}')
    print(f'Patches per image: {patches.shape[1]}')
    print(f'Elements per patch: {patches.shape[-1]}')
    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))

    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy().astype('uint8'))
        plt.axis('off')

    plt.show()


if __name__=="__main__":

    print('TensorFlow Version ' + tf.__version__)
    seed_everything()
    warnings.filterwarnings('ignore')

    model_name = "nofinetune"

    image_size = 224
    batch_size = 16
    n_classes = len(classes_list)

    learning_rate = 0.001
    weight_decay = 0.0001
    num_epochs = 200

    # each patch is patch_size x patch_size
    patch_size = 16  # 56  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    # la dimensione su cui vengono proiettati dall'encoder, quindi num_patches*projection_dim
    projection_dim = 1024  # 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 8
    mlp_head_units = [56, 28]

    # train_path = 'D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\Dataset\\FinalDataset\\Train\\'
    train_path = 'D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\Dataset\\Expression\\'
    test_path = 'D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\Dataset\\FinalDataset\\Test\\'

    # classes_list = ["YES", "NO"]
    classes_list = ["Happy", "Neutral"]

    X = []
    Y = []

    for c in classes_list:
        if os.path.isdir(os.path.join(train_path, c)):
            for file_name in glob.glob(os.path.join(train_path, c) + "//*.png"):
                image = cv2.imread(file_name, cv2.COLOR_BGR2RGB)
                if len(image.shape) < 3:
                    image = np.stack((image,) * 3, axis=-1)
                    print(image.shape)
                    print(file_name)

                image = cv2.resize(image, (image_size, image_size))
                X.append(image)
                y = [0] * len(classes_list)
                y[classes_list.index(c)] = 1
                Y.append(y)

    X = np.asarray(X)

    # show_patches(X)

    X = vit.preprocess_inputs(X)
    y = np.asarray(Y)

    X_t = []
    Y_t = []

    for c in classes_list:
        if os.path.isdir(os.path.join(test_path, c)):
            for file_name in glob.glob(os.path.join(test_path, c) + "//*.png"):
                image = cv2.imread(file_name, cv2.COLOR_BGR2RGB)

                if len(image.shape) < 3:
                    image = np.stack((image,) * 3, axis=-1)
                    print(image.shape)
                    print(file_name)

                image = cv2.resize(image, (image_size, image_size))
                X_t.append(image)
                y_t = [0] * len(classes_list)
                y_t[classes_list.index(c)] = 1
                Y_t.append(y_t)

    X_test = np.asarray(X_t)
    X_test = vit.preprocess_inputs(X_test)
    y_test = np.asarray(Y_t)

    X, X_valid, y, y_valid = sklearn.model_selection.train_test_split(X, y, test_size=0.15, random_state=1)

    decay_steps = np.shape(X)[0] // batch_size

    initial_learning_rate = learning_rate
    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate, decay_steps)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decayed_fn)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = vision_transformer()

    model.summary()

    tf.keras.utils.plot_model(model)

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                     min_delta=1e-4,
                                                     patience=50,
                                                     mode='max',
                                                     restore_best_weights=True,
                                                     verbose=1)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="D:\\" + model_name + "-{epoch:02d}.hdf5",
                                                      monitor='val_accuracy',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      mode='max')

    callbacks = [earlystopping, lr_scheduler, checkpointer]

    model.fit(x=X,
              y=y,
              validation_data=(X_valid, y_valid),
              batch_size=batch_size,
              epochs=num_epochs,
              callbacks=callbacks)

    print('Testing results')
    model.evaluate(X_test)
