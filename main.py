from model import CNN, ResNet, RIR
from utils import *
import tensorlayer as tl
import tensorflow as tf
import numpy as np

epoches = 1000       # Epoches
loss = {}       # Loss

if __name__ == '__main__':
    # Load data
    train_x, train_y, eval_x, eval_y, test_x, test_y = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
    train_y = to_categorical(train_y)
    print(np.shape(test_x))

    # Construct the network
    imgs_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    tags_ph = tf.placeholder(tf.float32, [None, 10])
    usual_cnn = CNN(imgs_ph)
    res_net = ResNet(imgs_ph)
    rir = RIR(imgs_ph)

    # Define loss and optimizer
    cnn_predict = usual_cnn()
    """
    resnet_predict = res_net()
    rir_predict = rir()
    """
    cnn_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tags_ph, logits=cnn_predict)
    """
    resnet_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tags_ph, logits=resnet_predict)
    rir_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tags_ph, logits=rir_predict)
    """
    cnn_loss_value = tf.reduce_mean(cnn_loss)

    """
    resnet_loss_value = tf.reduce_sum(resnet_loss)
    rir_loss_value = tf.reduce_sum(rir_loss)
    """

    cnn_train_op = tf.train.AdamOptimizer(0.0001).minimize(cnn_loss_value)
    """
    resnet_train_op = tf.train.AdamOptimizer().minimize(resnet_loss)
    rir_train_op = tf.train.AdamOptimizer().minimize(rir_loss)
    """

    # Train toward usual CNN
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoches):
            imgs_batch, label_batch = next_batch(train_x, train_y, batch_size=32)
            feed_dict = {
                imgs_ph: imgs_batch,
                tags_ph: label_batch
            }
            _cnn_loss = sess.run([cnn_loss_value,], feed_dict=feed_dict)
            _, = sess.run([cnn_train_op,], feed_dict=feed_dict)
            if i % 10 == 0:
                print('iter: ', i, '\tCNN loss: ', _cnn_loss)