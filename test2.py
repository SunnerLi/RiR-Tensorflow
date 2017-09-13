import tensorlayer as tl
import tensorflow as tf
import numpy as np

epoches = 2000       # Epoches

batch_index = 0
def next_batch(imgs, labels, batch_size=32):
    global batch_index
    if batch_index + batch_size >= np.shape(imgs)[0]:
        batch_index = -1 * batch_index
    batch_index += batch_size
    return imgs[batch_index:batch_index+batch_size, :], labels[batch_index:batch_index+batch_size, :]

def to_categorical(y, num_classes=None):
    """
        The implementation of to_categorical which is defined in Keras
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

if __name__ == '__main__':
    # Load data
    train_x, train_y, eval_x, eval_y, test_x, test_y = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
    train_y = to_categorical(train_y)

    # Construct the network
    imgs_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    tags_ph = tf.placeholder(tf.float32, [None, 10])

    # Softmax
    """    
    network = tl.layers.InputLayer(imgs_ph)
    network = tl.layers.DenseLayer(network, n_units = 512, act = tf.nn.tanh, name='cnn_fc1')
    network = tl.layers.DenseLayer(network, n_units = 256, act = tf.nn.tanh, name='cnn_fc2')
    network = tl.layers.DenseLayer(network, n_units = 128, act = tf.nn.tanh, name='cnn_fc3')
    network = tl.layers.DenseLayer(network, n_units = 10, act = tf.nn.softmax, name='cnn_fc4')
    """

    network = tl.layers.InputLayer(imgs_ph, name='cnn_input')
    network = tl.layers.Conv2dLayer(network, shape = [3, 3, 1, 8], act = tf.nn.relu, name='cnn_conv1')
    network = tl.layers.Conv2dLayer(network, shape = [3, 3, 8, 16], act = tf.nn.relu, name='cnn_conv2')
    network = tl.layers.BatchNormLayer(network, act = tf.nn.elu, name='cnn_bn_relu_1')
    network = tl.layers.MaxPool2d(network, name='cnn_maxpool1')
    network = tl.layers.Conv2dLayer(network, shape = [3, 3, 16, 32], act = tf.nn.relu, name='cnn_conv3')
    network = tl.layers.Conv2dLayer(network, shape = [3, 3, 32, 64], act = tf.nn.relu, name='cnn_conv4')
    network = tl.layers.BatchNormLayer(network, act = tf.nn.elu, name='cnn_bn_relu_2')
    network = tl.layers.MaxPool2d(network, name='cnn_maxpool2')
    network = tl.layers.FlattenLayer(network)
    network = tl.layers.DenseLayer(network, n_units = 256, act = tf.nn.tanh, name='cnn_fc1')
    network = tl.layers.DenseLayer(network, n_units = 10, act = tf.nn.softmax, name='cnn_fc2')
    



    # Define loss and optimizer
    cnn_predict = network.outputs
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tags_ph, logits=cnn_predict))
    cnn_train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # Train toward usual CNN
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoches):
            imgs_batch, label_batch = next_batch(train_x, train_y, batch_size=32)
            feed_dict = {
                imgs_ph: imgs_batch,
                tags_ph: label_batch
            }
            _cnn_loss = sess.run([cross_entropy], feed_dict=feed_dict)
            _, = sess.run([cnn_train_op], feed_dict=feed_dict)
            if i % 10 == 0:
                print('iter: ', i, '\tCNN loss: ', _cnn_loss)