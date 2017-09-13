from utils import *
import tensorlayer as tl
import tensorflow as tf
import numpy as np

epoches = 1000       # Epoches

if __name__ == '__main__':
    # Load data
    train_x, train_y, eval_x, eval_y, test_x, test_y = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
    train_y = to_categorical(train_y)

    # Construct the network
    imgs_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    tags_ph = tf.placeholder(tf.float32, [None, 10])

    # 1 & 2 conv
    network = tl.layers.InputLayer(imgs_ph, name='cnn_input')
    """
    network = tl.layers.Conv2d(network, n_filter=32, name='cnn_conv1')
    network = tl.layers.Conv2d(network, n_filter=32, name='cnn_conv2')
    """
    network = tl.layers.Conv2dLayer(network, shape = [3, 3, 1, 32], act = tf.nn.relu, name='cnn_conv1')
    network = tl.layers.Conv2dLayer(network, shape = [3, 3, 32, 64], act = tf.nn.relu, name='cnn_conv2')

    network = tl.layers.BatchNormLayer(network, act = tf.nn.elu, name='cnn_bn_relu_1')
    network = tl.layers.MaxPool2d(network, name='cnn_maxpool1')

    # 3 & 4 conv
    network = tl.layers.Conv2dLayer(network, shape = [3, 3, 64, 128], act = tf.nn.relu, name='cnn_conv3')
    network = tl.layers.Conv2dLayer(network, shape = [3, 3, 128, 256], act = tf.nn.relu, name='cnn_conv4')
    network = tl.layers.BatchNormLayer(network, act = tf.nn.elu, name='cnn_bn_relu_2')
    network = tl.layers.MaxPool2d(network, name='cnn_maxpool2')

    # Softmax
    network = tl.layers.FlattenLayer(network)
    network = tl.layers.DenseLayer(network, n_units = 256, act = tf.nn.tanh, name='cnn_fc1')
    network = tl.layers.DenseLayer(network, n_units = 10, act = tf.nn.softmax, name='cnn_fc2')

    # Define loss and optimizer
    cnn_predict = network.outputs
    cnn_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tags_ph, logits=cnn_predict)
    cnn_loss_value = tf.reduce_sum(cnn_loss)
    cnn_train_op = tf.train.AdamOptimizer(0.01).minimize(cnn_loss)

    # Train toward usual CNN
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoches):
            imgs_batch, label_batch = next_batch(train_x, train_y, batch_size=32)
            feed_dict = {
                imgs_ph: imgs_batch,
                tags_ph: label_batch
            }
            _cnn_loss = sess.run([cnn_loss_value], feed_dict=feed_dict)
            _, = sess.run([cnn_train_op], feed_dict=feed_dict)
            if i % 10 == 0:
                print('iter: ', i, '\tCNN loss: ', _cnn_loss)