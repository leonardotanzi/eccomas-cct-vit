from TrainViT import seed_everything
import warnings
import tensorflow as tf
import os
import glob
import cv2
import numpy as np
from vit_keras import vit
import sklearn
from tensorflow.keras import layers
import tensorflow_addons as tfa


class ClassToken(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)


class AddPositionEmbs(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = tf.keras.layers.Dense(hidden_size, name="query")
        self.key_dense = tf.keras.layers.Dense(hidden_size, name="key")
        self.value_dense = tf.keras.layers.Dense(hidden_size, name="value")
        self.combine_heads = tf.keras.layers.Dense(hidden_size, name="out")

    # pylint: disable=no-self-use
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights


# pylint: disable=too-many-instance-attributes
class TransformerBlock(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0",
                ),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False)
                )
                if hasattr(tf.keras.activations, "gelu")
                else tf.keras.layers.Lambda(
                    lambda x: tfa.activations.gelu(x, approximate=False)
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

    def get_config(self):
        return {
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
        }

def vision_transformer(image_size, patch_size, num_heads, hidden_size, num_layers, mlp_dim, dropout, name):

    x = layers.Input(shape=(image_size, image_size, 3))
    patches = tf.keras.layers.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="embedding",
    )(x)
    y = layers.Reshape((patches.shape[1] * patches.shape[2], hidden_size))(patches)
    # credo aggiunga semplicemente un vettore di zeri
    y = ClassToken(name="class_token")(y)
    # somma (non è una concatenazione) dei valori random al layer di input
    y = AddPositionEmbs(name="Transformer/posembed_input")(y)
    for n in range(num_layers):
        y, _ = TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer/encoderblock_{n}",
        )(y)
    y = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(y)
    y = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(y)

    y = layers.Flatten()(y)
    y = layers.Dense(4096, activation=tf.nn.gelu)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.5)(y)
    y = layers.Dense(len(classes_list), 'softmax')(y)

    return tf.keras.models.Model(inputs=x, outputs=y, name=name)


if __name__=="__main__":

    print('TensorFlow Version ' + tf.__version__)
    seed_everything()
    warnings.filterwarnings('ignore')

    model_name = "nofinetune"
    classes_list = ["YES", "NO"]
    # classes_list = ["Happy", "Neutral"]

    image_size = 224
    batch_size = 16
    n_classes = len(classes_list)
    learning_rate = 0.01
    weight_decay = 0.0001
    num_epochs = 200
    # each patch is patch_size x patch_size
    patch_size = 16  # 56  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    # la dimensione su cui vengono proiettati dall'encoder, quindi num_patches*projection_dim

    train_path = 'D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\Dataset\\FinalDataset\\Train\\'
    # train_path = 'D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\Dataset\\Expression\\'
    test_path = 'D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\Dataset\\FinalDataset\\Test\\'

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

    model = vision_transformer(image_size, patch_size, num_heads=16, hidden_size=768, num_layers=24, mlp_dim=4096,
                               dropout=0.1, name="vit-l16")

    model.summary()

    learning_rate = 1e-4
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
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

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=model_name + "-{epoch:02d}.hdf5",
                                                      monitor="val_accuracy",
                                                      verbose=1,
                                                      save_best_only=False,
                                                      save_weights_only=False,
                                                      mode="max",
                                                      period=10)

    callbacks = [earlystopping, checkpointer, reduce_lr]  # WarmupExponentialDecay(lr_base=0.0002, decay=0, warmup_epochs=2)]

    model.fit(x=X,
              y=y,
              batch_size=batch_size,
              epochs=num_epochs,
              validation_data=(X_valid, y_valid),
              # class_weight=class_weights_train,
              callbacks=callbacks)

    model.save(model_name + ".hdf5")