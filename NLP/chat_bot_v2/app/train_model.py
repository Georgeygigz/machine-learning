
import tflearn
import tensorflow.compat.v1 as tf
from .read_file import read_file
tf.disable_v2_behavior()

training, output,a,c,b =  read_file()


def train_model():
    tf.reset_default_graph()
    network = tflearn.input_data(shape=[None, len(training[0])])
    network = tflearn.fully_connected(network,8)
    network = tflearn.fully_connected(network,8)
    network = tflearn.fully_connected(
        network,len(output[0]), activation="softmax")
    network = tflearn.regression(network)

    model =tflearn.DNN(network)

    try:
        model.load("model.tflearn")
    except:
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model.tflearn")

    return model


