from visualize import draw, meanError
from model import CNN, ResNet, RiR
from utils import *
import tensorlayer as tl
import tensorflow as tf
import numpy as np

epoches = 1                 # Epoches
iters = 400                 # Iterators
M = 2                       # Monte-Carlo liked scalar
record = {
    'loss': {               # Loss
        'cnn': [],
        'resnet': [],
        'rir': []
    },
    'acc': {                # Accuracy
        'cnn': [],
        'resnet': [],
        'rir': []
    }
}

def recordTrainResult(cnn_loss, res_net_loss, rir_loss, cnn_acc, res_net_acc, rir_acc):
    """
        Append the training result into the corresponding list

        Arg:    cnn_loss        - The loss value of usual CNN
                res_net_loss    - The loss value of ResNet
                rir_loss        - The loss value of ResNet in ResNet Network
                cnn_acc         - The accuracy value of usual CNN
                res_net_acc     - The accuracy value of ResNet
                rir_acc         - The accuracy value of ResNet in ResNet Network
                
    """
    global record
    record['loss']['cnn'].append(cnn_loss)
    record['loss']['resnet'].append(res_net_loss)
    record['loss']['rir'].append(rir_loss) 
    record['acc']['cnn'].append(cnn_acc)
    record['acc']['resnet'].append(res_net_acc)
    record['acc']['rir'].append(rir_acc)

if __name__ == '__main__':
    # Load data
    train_x, train_y, eval_x, eval_y, test_x, test_y = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
    #train_x, train_y, test_x, test_y = tl.files.load_cifar10_dataset()
    train_x -= 0.5
    #train_x = (train_x - 127.5) / 127.5
    train_y = to_categorical(train_y)

    # Construct the network
    imgs_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    tags_ph = tf.placeholder(tf.float32, [None, 10])
    usual_cnn = CNN(imgs_ph, tags_ph)
    res_net = ResNet(imgs_ph, tags_ph)
    rir = RiR(imgs_ph, tags_ph)

    # Train toward usual CNN
    with tf.Session() as sess:
        for i in range(M):
            sess.run(tf.global_variables_initializer())
            print('Scalar: ', i)
            for j in range(epoches):
                for k in range(iters):
                    imgs_batch, label_batch = next_batch(train_x, train_y, batch_size=32)
                    feed_dict = {
                        imgs_ph: imgs_batch,
                        tags_ph: label_batch
                    }
                    _cnn_loss, _cnn_acc, _ = sess.run([usual_cnn.loss, usual_cnn.accuracy, usual_cnn.optimize], feed_dict=feed_dict)
                    _res_net_loss, _res_net_acc, _ = sess.run([res_net.loss, res_net.accuracy, res_net.optimize], feed_dict=feed_dict)
                    _rir_loss, _rir_acc, _ = sess.run([rir.loss, rir.accuracy, rir.optimize], feed_dict=feed_dict)
                    if k % 10 == 0:
                        print('iter: ', k, '\tCNN loss: ', _cnn_loss, '\tacc: ', _cnn_acc, '\tResNet loss: ', _res_net_loss, \
                            '\tacc: ', _res_net_acc, '\tRiR loss: ', _rir_loss, '\tacc: ', _rir_acc)
                        recordTrainResult(_cnn_loss, _res_net_loss, _rir_loss, _cnn_acc, _res_net_acc, _rir_acc)

    # Visualize
    record = meanError(record, M)
    draw(record)