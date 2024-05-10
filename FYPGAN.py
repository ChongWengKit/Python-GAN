import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from keras import Model, Input
from keras.applications import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.layers import Concatenate, Activation, GlobalAveragePooling2D
from keras.losses import MeanSquaredError,BinaryCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import array_to_img
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D, BatchNormalization

print("TensorFlow version:", tf.__version__)
datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=(0.8, 1.2),
            zoom_range=0.2
        )
processed_images = 0
data_size = 64*100
base_path = 'D:/Birds64'
pickle_file_path = 'D:/word2vec_train_bird_test_300.pickle'
file_path = 'D:/output_train_bird.txt'
data_dir="D:/Birds64"
x_train = []
y_train=[]
first_3_vectors = []
batch_size=64

def check_tensorflow_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')

    if len(physical_devices) == 0:
        print("No GPUs found.")
    else:
        print("Available GPU(s):")
        for device in physical_devices:
            print(f"  - {device.name}")


# Read your text file
with open(pickle_file_path, 'rb') as file:
    y_train = pickle.load(file)

if 'embeddings' in y_train:
    embeddings_data = y_train['embeddings'][:data_size]
    y_train = np.array(embeddings_data)
else:
    print("Error: 'embeddings' key not found in the dictionary.")


with open(file_path, 'r') as file:
    lines = file.readlines()
image_filenames = [line.split('|')[0].strip() for line in lines]
for image_filename in image_filenames:
    if processed_images >= data_size:
        break
    if image_filename and image_filename.strip() and image_filename.lower().endswith(".jpg"):
        image_path = os.path.join(base_path, image_filename)
        image_path = os.path.normpath(image_path)
        image = Image.open(image_path)
        if image.size == (64, 64):
            if image.mode == 'L':
                image = image.convert('RGB')
        image_data = image
        image_data = (np.array(image_data) / 127.5) - 1.0

        x_train.append(image_data)
        processed_images += 1




    else:
        print("empty?" + image_filename)


x_train = np.array(x_train)
print(x_train.shape)
print(y_train.shape)

dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
for i, (x, y) in enumerate(dataset):
    if(i == 0):
        first_3_vectors.append(y)
    if(i == 2000):
        first_3_vectors.append(y)

    if(i== 4000):
       first_3_vectors.append(y)
       break

concatenated_vectors = tf.stack(first_3_vectors, axis=0)

buffer_size = len(x_train)
dataset = dataset.shuffle(buffer_size)
dataset=dataset.batch(batch_size)

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "fyp_gan_checkpoint.h5"),
    save_best_only=True,
    save_weights_only=True,
    monitor="g_loss",
    verbose=1,
)
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.d_model = d_model

    def build(self, input_shape):
        self.W_q = self.add_weight("W_q", shape=(self.d_model, self.d_model))
        self.W_k = self.add_weight("W_k", shape=(self.d_model, self.d_model))
        self.W_v = self.add_weight("W_v", shape=(self.d_model, self.d_model))

    def call(self, inputs):
        q = tf.matmul(inputs, self.W_q)
        k = tf.matmul(inputs, self.W_k)
        v = tf.matmul(inputs, self.W_v)

        attn_scores = tf.matmul(q, k, transpose_b=True)
        attn_scores /= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attn_output = tf.matmul(attn_weights, v)

        return attn_output

def build_generator():
  noise_dim = 100
  embedding_dim = 200
  inputs = Input(shape=(noise_dim,))
  embedding_input = Input(shape=(embedding_dim,))
  combined_input = Concatenate()([inputs, embedding_input])

  x = Dense(128* 8 * 8)(combined_input)
  x = Reshape((8, 8, 128))(x)

  x = Conv2D(128, (5, 5), padding='same')(x)  # Reduced filter count from 128
  x = LeakyReLU(alpha=0.1)(x)
  x = SelfAttention(128)(x)

  x = UpSampling2D((2, 2))(x)
  x = Conv2D(64, (3, 3), padding='same')(x)  # Reduced filter count from 64
  x = LeakyReLU(alpha=0.1)(x)

  x = UpSampling2D((2, 2))(x)
  x = Conv2D(64, (3, 3), padding='same')(x)  # Reduced filter count from 64
  x = LeakyReLU(alpha=0.1)(x)

  x = Conv2D(64, (3, 3), padding='same')(x)  # Reduced filter count from 64
  x = LeakyReLU(alpha=0.1)(x)

  x = UpSampling2D((2, 2))(x)
  x = Conv2D(32, (3, 3), padding='same')(x)  # Reduced filter count from 32
  x = LeakyReLU(alpha=0.1)(x)

  x = Conv2D(32, (3, 3), padding='same')(x)
  x = LeakyReLU(alpha=0.1)(x)

  x = Conv2D(3, (3, 3), padding='same')(x)
  generated_image = Activation('tanh')(x)

  model = Model(inputs=[inputs, embedding_input], outputs=generated_image, name='Generator')
  return model
generator = build_generator()
generator.summary()

def build_discriminator():
    embedding_dim=200
    inputs = Input(shape=(64, 64, 3))
    embedding_input = Input(shape=(embedding_dim,))
    x = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, (3,3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, (3,3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x=SelfAttention(256)(x)

    x = GlobalAveragePooling2D()(x)  # Maintains spatial information
    combined_input=Concatenate()([x,embedding_input])
    x = Dense(128)(combined_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Dense(1, activation='sigmoid')(x)
    discriminator = Model(inputs=[inputs, embedding_input], outputs=output, name='Discriminator')
    return discriminator

discriminator = build_discriminator()
discriminator.summary()
g_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
d_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
g_loss = MeanSquaredError()
d_loss = BinaryCrossentropy()
class GAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
    def compile(self, g_opt, d_opt, g_loss, d_loss,clip_value=1.0, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.clip_value = clip_value  # Store clip_value in the model

    def train_step(self, batch):
        real_images, real_words = batch
        real_images= tf.cast(real_images,tf.float32)
        noise = tf.cast(tf.random.normal((batch_size,100)),tf.float32)
        fake_images = self.generator([noise,real_words], training=False)
        with tf.GradientTape() as disc_tape:
            d_real = self.discriminator([real_images,real_words], training=True)
            d_fake = self.discriminator([fake_images,real_words], training=True)
            d_realfake_l = tf.concat([d_real, d_fake], axis=0)
            d_realfake = tf.concat([tf.ones_like(d_real)*0.9, tf.ones_like(d_fake) * 0.1], axis=0)
            d_loss = self.d_loss(d_realfake, d_realfake_l)
        d_grad = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.d_opt.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))
        with tf.GradientTape() as generator_tape:
            noise = tf.cast(tf.random.normal((batch_size,100)), tf.float32)
            gen_images = self.generator([noise,real_words], training=True)
            predicted_labels = self.discriminator([gen_images,real_words], training=False)
            ones_like_labels = tf.ones_like(predicted_labels)*0.9
            g_loss = self.g_loss(ones_like_labels, predicted_labels)
        g_grad = generator_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grad, self.generator.trainable_variables))
        return {"d_loss": d_loss, "g_loss": g_loss}

gan = GAN(generator, discriminator)
gan.compile(g_opt, d_opt, g_loss, d_loss)
def plot(hist):
    plt.suptitle('Loss')
    plt.plot(hist.history['d_loss'], label='d_loss')
    plt.plot(hist.history['g_loss'], label='g_loss')
    plt.legend()
    plt.show()

class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=100,inception_frequency=10):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.inception_frequency = inception_frequency

    def on_epoch_end(self,epoch,logs=None):

        random_latent_vectors = tf.random.normal((self.num_img, self.latent_dim))
        generated_images = self.model.generator([random_latent_vectors,concatenated_vectors])
        print(generated_images.numpy())
        generated_images = (generated_images + 1) / 2.0
        generated_images = (generated_images * 255)
        generated_images = tf.cast(generated_images, tf.uint8)
        generated_images.numpy()
        for i in range(3):
            img = array_to_img(generated_images[i])
            img.save(os.path.join('images', f'{epoch}_{i}.png'))
class LossPlotter(Callback):
    def __init__(self,figsize):
        super(LossPlotter, self).__init__()
        self.d_losses = []
        self.g_losses = []
        self.fig, self.ax = plt.subplots(figsize=figsize)
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        d_loss = logs.get('d_loss')
        g_loss = logs.get('g_loss')
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)

        self.ax.clear()
        self.ax.plot(range(epoch + 1), self.d_losses, label='Discriminator Loss', color='blue')
        self.ax.plot(range(epoch + 1), self.g_losses, label='Generator Loss', color='orange')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('GAN Training Loss')
        self.ax.legend()
        plt.pause(0.01)


monitor_callback = ModelMonitor(num_img=3, latent_dim=100,inception_frequency=10)
plotter_callback = LossPlotter(figsize=(3, 2))

hist = gan.fit(dataset, epochs=500, callbacks=[monitor_callback,plotter_callback,checkpoint_callback])


