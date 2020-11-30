"""
The script demonstrates a simple example of using ART with Keras. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
# %%
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
from keras import backend as K
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from art.attacks.evasion import FastGradientMethod, DeepFool
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
   try:
       # Currently, memory growth needs to be the same across GPUs
       for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
       logical_gpus = tf.config.experimental.list_logical_devices('GPU')
       print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
   except RuntimeError as e:
       # Memory growth must be set before GPUs have been initialized
       print(e)
else:
    print("Not using GPU!!!!!")

# %%
def plot_examples(examples):
    """
    examples[i][j]: [[(orig, adv, ex),(),(),(),()],[],[],...] 
    """
    cnt = 0
    fig = plt.figure(figsize=(8,10))
    for i in range(len(examples)):
        cnt += 1
        plt.subplot(1,len(examples),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(examples[i],cmap='gray')
    plt.tight_layout()
    plt.show()
    plt.close(fig)

# Step 1: Load the MNIST dataset
# %%
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 2: Create the model

model = Sequential()
model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(23, 23, 4)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(
    loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=["accuracy"]
)

# Step 3: Create the ART classifier

classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)

# Step 4: Train the ART classifier
# %%
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)
# %%
# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
# %%
# Step 6: Generate adversarial test examples
attack = DeepFool(classifier)
x_test_adv = attack.generate(x=x_test)
# %%
# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
#%%
plot_examples([x_test_adv[3], x_test_adv[8], x_test_adv[12], x_test_adv[18]])
# %%
K.clear_session()
