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

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 200
image_size = 224
num_classes = 7
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
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
        ],
    )

    checkpoint_filepath = "Checkpoint\\"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.2,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history


if __name__ == "__main__":

    # (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    #     #
    #     # y_train = keras.utils.to_categorical(y_train, num_classes)
    #     # y_test = keras.utils.to_categorical(y_test, num_classes)
    # Note the rescaling layer. These layers have pre-defined inference behavior.

    # train_path = 'D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\Dataset\\FinalDataset\\Train\\'
    # test_path = 'D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\Dataset\\FinalDataset\\Test\\'
    # #
    # # train_path = "D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\Dataset\\Expression\\"
    # # test_path = "D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\Dataset\\Expression\\"
    #
    # classes_list = ["YES", "NO"]

    train_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Train\\'
    test_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\'

    classes_list = ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"]
    # classes_list = ["B", "Unbroken"]

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

    x_train = np.asarray(X)  # / 255.0
    y_train = np.asarray(Y)
    x_train, y_train = shuffle(x_train, y_train, random_state=0)

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

    x_test = np.asarray(X_t)  # / 255.0
    y_test = np.asarray(Y_t)
    x_test, y_test = shuffle(x_test, y_test, random_state=0)

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    cct_model = create_cct_model()
    history = run_experiment(cct_model)

    print(cct_model.evaluate(x_test, y_test, batch_size=batch_size))

    predicted_classes = np.argmax(cct_model.predict(x_test), axis=1)
    true_classes = np.argmax(y_test, axis=1)
    # class_labels = list(test_gen.class_indices.keys())

    confusionmatrix = confusion_matrix(true_classes, predicted_classes)
    print(confusionmatrix)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)
    plt.show()

    # confusionmatrix_norm = confusionmatrix / confusionmatrix.astype(np.float).sum(axis=1)
    # confusionmatrix_norm.round(decimals=2)
    confusionmatrix_norm = np.around(confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis],
                                     decimals=2)
    print(confusionmatrix_norm)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix_norm, cmap='Blues', annot=True, cbar=True)
    plt.show()

    print(classification_report(true_classes, predicted_classes))