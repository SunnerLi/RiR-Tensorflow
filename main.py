from visualize import draw, meanError
from model import CNN, ResNet, RiR
from utils import *
import tensorlayer as tl
import tensorflow as tf
import numpy as np

epoches = 200       # Epoches
M = 5               # Monte-Carlo liked scalar
loss = {            # Loss
    'cnn': [],
    'resnet': [],
    'rir': []
}           

if __name__ == '__main__':
    # Load data
    train_x, train_y, eval_x, eval_y, test_x, test_y = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
    train_x -= 0.5
    train_y = to_categorical(train_y)

    # Construct the network
    imgs_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    tags_ph = tf.placeholder(tf.float32, [None, 10])
    usual_cnn = CNN(imgs_ph, tags_ph)
    res_net = ResNet(imgs_ph, tags_ph)
    rir = RiR(imgs_ph, tags_ph)

    # Train toward usual CNN
    with tf.Session() as sess:
        for j in range(M):
            sess.run(tf.global_variables_initializer())
            print('Scalar: ', j)
            for i in range(epoches):
                imgs_batch, label_batch = next_batch(train_x, train_y, batch_size=32)
                feed_dict = {
                    imgs_ph: imgs_batch,
                    tags_ph: label_batch
                }
                _cnn_loss, _cnn_acc, _ = sess.run([usual_cnn.loss, usual_cnn.accuracy, usual_cnn.optimize], feed_dict=feed_dict)
                _res_net_loss, _res_net_acc, _ = sess.run([res_net.loss, res_net.accuracy, res_net.optimize], feed_dict=feed_dict)
                _rir_loss, _rir_acc, _ = sess.run([rir.loss, rir.accuracy, rir.optimize], feed_dict=feed_dict)
                if i % 10 == 0:
                    print('iter: ', i, '\tCNN loss: ', _cnn_loss, '\tacc: ', _cnn_acc, '\tResNet loss: ', _res_net_loss, \
                        '\tacc: ', _res_net_acc, '\tRiR loss: ', _rir_loss, '\tacc: ', _rir_acc)
                    loss['cnn'].append(_cnn_loss)
                    loss['resnet'].append(_res_net_loss)
                    loss['rir'].append(_rir_loss)

    # Visualize
    loss = meanError(loss, M)
    draw(loss)