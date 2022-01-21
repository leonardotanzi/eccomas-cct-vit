from tensorflow.keras import layers
from tensorflow import keras
import glob
import cv2
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split


positional_emb = True
conv_layers = 2
projection_dim = 128

num_heads = 2
transformer_units = [
    projection_dim,
    projection_dim,
]
transformer_layers = 2
stochastic_depth_rate = 0.1

classes_list = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14"]

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 1
image_size = 224
num_classes = len(classes_list)
input_shape = (224, 224, 3)


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class CCTTokenizer(layers.Layer):
    def __init__(
        self,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        num_conv_layers=conv_layers,
        num_output_channels=[64, 128],
        positional_emb=positional_emb,
        **kwargs,
    ):
        super(CCTTokenizer, self).__init__(**kwargs)

        # This is our tokenizer.
        self.conv_model = keras.Sequential()
        for i in range(num_conv_layers):
            self.conv_model.add(
                layers.Conv2D(
                    num_output_channels[i],
                    kernel_size,
                    stride,
                    padding="valid",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            self.conv_model.add(layers.ZeroPadding2D(padding))
            self.conv_model.add(
                layers.MaxPool2D(pooling_kernel_size, pooling_stride, "same")
            )

        self.positional_emb = positional_emb

    def call(self, images):
        outputs = self.conv_model(images)
        # After passing the images through our mini-network the spatial dimensions
        # are flattened to form sequences.
        reshaped = tf.reshape(
            outputs,
            (-1, tf.shape(outputs)[1] * tf.shape(outputs)[2], tf.shape(outputs)[-1]),
        )
        return reshaped

    def positional_embedding(self, image_size):
        # Positional embeddings are optional in CCT. Here, we calculate
        # the number of sequences and initialize an `Embedding` layer to
        # compute the positional embeddings later.
        if self.positional_emb:
            dummy_inputs = tf.ones((1, image_size, image_size, 3))
            dummy_outputs = self.call(dummy_inputs)
            sequence_length = tf.shape(dummy_outputs)[1]
            projection_dim = tf.shape(dummy_outputs)[-1]

            embed_layer = layers.Embedding(
                input_dim=sequence_length, output_dim=projection_dim
            )
            return embed_layer, sequence_length
        else:
            return None


def create_cct_model(image_size=image_size, input_shape=input_shape, num_heads=num_heads, projection_dim=projection_dim, transformer_units=transformer_units):

    inputs = layers.Input(input_shape)

    # # Augment data.
    # augmented = data_augmentation(inputs)

    # Encode patches.
    cct_tokenizer = CCTTokenizer()
    encoded_patches = cct_tokenizer(inputs)

    # Apply positional embedding.
    if positional_emb:
        pos_embed, seq_length = cct_tokenizer.positional_embedding(image_size)
        positions = tf.range(start=0, limit=seq_length, delta=1)
        position_embeddings = pos_embed(positions)
        encoded_patches += position_embeddings

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    attention_weights = tf.nn.softmax(layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    weighted_representation = tf.squeeze(weighted_representation, -2)

    # Classify outputs.
    logits = layers.Dense(num_classes)(weighted_representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    checkpoint_filepath = "..\\Checkpoints\\"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.2,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(X_val, y_val)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history


if __name__ == "__main__":

    dataset_path = "..\\..\\10-01-22_database_Aiello"

    # datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2, horizontal_flip=True, preprocessing_function=preprocess_input)
    #
    # valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocess_input)
    #
    # train_gen = datagen.flow_from_directory(directory=train_path, subset="training", classes=classes_list, batch_size=batch_size,
    #                                         seed=1, color_mode='rgb', shuffle=True, class_mode='categorical', target_size=(image_size, image_size))
    #
    # valid_gen = valid_datagen.flow_from_directory(directory=train_path, subset='validation', classes=classes_list, batch_size=batch_size,
    #                                           seed=1, color_mode='rgb', shuffle=False, class_mode='categorical', target_size=(image_size, image_size))

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

                image = cv2.resize(image, (image_size, image_size))
                X.append(image)
                y = [0] * len(classes_list)
                y[classes_list.index(c)] = 1
                Y.append(y)

    X_full = np.asarray(X)
    y_full = np.asarray(Y)

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.1, random_state=1, shuffle=True)

    print(f"x_train shape: {X_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {X_test.shape} - y_test shape: {y_test.shape}")

    cct_model = create_cct_model()
    optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)

    cct_model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    checkpoint_filepath = "Checkpoints\\CCT\\"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = cct_model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.2,
        callbacks=[checkpoint_callback],
    )

    cct_model.load_weights(checkpoint_filepath)
    _, accuracy = cct_model.evaluate(X_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    print(cct_model.evaluate(X_test, y_test, batch_size=batch_size))

    predicted_classes = np.argmax(cct_model.predict(X_test), axis=1)
    true_classes = np.argmax(y_test, axis=1)
    # class_labels = list(test_gen.class_indices.keys())

    confusionmatrix = confusion_matrix(true_classes, predicted_classes)
    print(confusionmatrix)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)
    plt.show()

    confusionmatrix_norm = np.around(confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis],
                                     decimals=2)
    print(confusionmatrix_norm)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix_norm, cmap='Blues', annot=True, cbar=True)
    plt.show()

    print(classification_report(true_classes, predicted_classes))