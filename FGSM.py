from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import autokeras as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
import random, numpy as np

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

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

img_rows, img_cols, channels = 28, 28, 1

model = load_model("model_vanilla_mnist")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_classes = 10

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape((-1, img_rows, img_cols, channels))
x_test = x_test.reshape((-1, img_rows, img_cols, channels))

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

def plot_accuracies(epsilons, accuracies):
    fig = plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, epsilons[-1] + 0.05 , step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.savefig('foo.png')
    plt.show()
    plt.close(fig)

def plot_examples(epsilons, examples):
    """
    examples[i][j]: [[(orig, adv, ex),(),(),(),()],[],[],...] 
    """
    cnt = 0
    fig = plt.figure(figsize=(8,10))
    print(examples)
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {:.2f}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.savefig('foo2.png')
    plt.show()
    plt.close(fig)

def FGSM(points=10):
    from art.attacks.evasion import FastGradientMethod
    from art.estimators.classification import TensorFlowV2Classifier


    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    classifier = TensorFlowV2Classifier(model=model, nb_classes=10, input_shape=(28, 28, 1), loss_object=loss_object, 
                clip_values=(0, 1), channels_first=False)

    # Craft adversarial samples with FGSM
    epsilons = [0.05 * i for i in range(points)]  # Maximum perturbation
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy on normal sample: %.2f%% eps: %.2f" % (acc * 100, 0))
    accuracies = [acc]
    examples = []
    for epsilon in epsilons[1:]:
        adv_crafter = FastGradientMethod(classifier, eps=epsilon)
        x_test_adv = adv_crafter.generate(x=x_test)

        # Evaluate the classifier on the adversarial examples
        preds = np.argmax(classifier.predict(x_test_adv), axis=1)
        acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print("\nTest accuracy on adversarial sample: %.2f%% eps: %.2f" % (acc * 100, epsilon))
        accuracies.append(acc)
        example = []
        preds = np.argmax(classifier.predict(x_test_adv), axis=1)
        labels = np.argmax(y_test, axis=1)
        for i in range(len(preds)):
            p, l = preds[i], labels[i]
            if p != l:
                orig = l
                adv = p
                ex = x_test_adv[i]
                example.append((orig, adv, ex))
            if len(example) == 5:
                break
        examples.append(example)
    plot_accuracies(epsilons, accuracies)
    plot_examples(epsilons[1:], examples)

    


def Deepfool(points=2, steps=0.05):
    from art.attacks.evasion import NewtonFool
    from art.estimators.classification import TensorFlowV2Classifier

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    classifier = TensorFlowV2Classifier(model=model, nb_classes=10, input_shape=(28, 28, 1), loss_object=loss_object, 
                clip_values=(0, 1), channels_first=False)
    
    # Craft adversarial samples with FGSM
    epsilons = [0.2 * i + 0.1 for i in range(points)]  # Maximum perturbation
    preds = np.argmax(classifier.predict(x_test[:1000]), axis=1)
    acc = np.sum(preds == np.argmax(y_test[:1000], axis=1)) / y_test[:1000].shape[0]
    print("\nTest accuracy on normal sample: %.2f%% eps: %.2f" % (acc * 100, 0))
    accuracies = [acc]
    examples = []
    for epsilon in epsilons[1:]:
        adv_crafter = NewtonFool(classifier)
        x_test_adv = adv_crafter.generate(x=x_test[:1000], y=y_test[:1000])

        # Evaluate the classifier on the adversarial examples
        preds = np.argmax(classifier.predict(x_test_adv), axis=1)
        acc = np.sum(preds == np.argmax(y_test[:1000], axis=1)) / y_test[:1000].shape[0]
        print("\nTest accuracy on adversarial sample: %.2f%% eps: %.2f" % (acc * 100, epsilon))
        accuracies.append(acc)
        example = []
        preds = np.argmax(classifier.predict(x_test_adv), axis=1)
        labels = np.argmax(y_test[:1000], axis=1)
        for i in range(len(preds)):
            p, l = preds[i], labels[i]
            if p != l:
                orig = l
                adv = p
                ex = x_test_adv[i]
                example.append((orig, adv, ex))
            if len(example) == 5:
                break
        examples.append(example)
    plot_accuracies(epsilons, accuracies)
    plot_examples(epsilons[1:], examples)

Deepfool()