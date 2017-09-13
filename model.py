import tensorlayer as tl
import tensorflow as tf

class Net(object):
    def work(self, imgs_ph, tags_ph, predict):
        # Create the wrapper
        self.predict = predict
        self.tags_ph = tags_ph

        # Others
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.tags_ph, logits=self.predict))
        self.optimize = tf.train.AdamOptimizer().minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast( \
            tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.tags_ph, 1)), "float"), name='accuracy')

class CNN(Net):
    def __init__(self, imgs_ph, tags_ph):
        self.imgs_ph = imgs_ph
        self.tags_ph = tags_ph

        # 1st conv fix channel
        self.network = tl.layers.InputLayer(self.imgs_ph, name='cnn_input')
        self.network = tl.layers.Conv2d(self.network, n_filter=16, act = tf.nn.relu, name='cnn_conv1')

        # 2 & 3 conv
        self.network = tl.layers.Conv2d(self.network, n_filter=16, name='cnn_conv2')
        self.network = tl.layers.Conv2d(self.network, n_filter=16, name='cnn_conv3')
        self.network = tl.layers.BatchNormLayer(self.network, act = tf.nn.relu, name='cnn_bn_relu_1')
        self.network = tl.layers.MaxPool2d(self.network, name='cnn_maxpool1')

        # 4th conv fix channel
        self.network = tl.layers.Conv2d(self.network, n_filter=32, act = tf.nn.relu, name='cnn_conv4')

        # 5 & 6 conv
        self.network = tl.layers.Conv2d(self.network, n_filter=32, name='cnn_conv5')
        self.network = tl.layers.Conv2d(self.network, n_filter=32, name='cnn_conv6')
        self.network = tl.layers.BatchNormLayer(self.network, act = tf.nn.relu, name='cnn_bn_relu_2')
        self.network = tl.layers.MaxPool2d(self.network, name='cnn_maxpool2')

        # Softmax
        self.network = tl.layers.Conv2d(self.network, n_filter=64, act = tf.nn.relu, name='cnn_conv7')
        self.network = tl.layers.FlattenLayer(self.network, name='cnn_flat')
        self.network = tl.layers.DenseLayer(self.network, n_units = 10, act = tf.nn.softmax, name='cnn_fc')
        self.predict = self.network.outputs
        self.work(imgs_ph, tags_ph, self.predict)

class ResNet(Net):
    def __init__(self, imgs_ph, tags_ph):
        self.imgs_ph = imgs_ph
        self.tags_ph = tags_ph

        # Conv fix channel
        self.network = tl.layers.InputLayer(self.imgs_ph, name='resnet_input')
        self.input_layer = tl.layers.Conv2d(self.network, n_filter=16, act = tf.nn.relu, name='resnet_conv1')

        # -----------------------------
        # 1st general residual block
        # -----------------------------
        self.network = tl.layers.Conv2d(self.input_layer, n_filter=16, name='resnet_conv2')
        self.network = tl.layers.Conv2d(self.network, n_filter=16, name='resnet_conv3')
        self.network = tl.layers.ElementwiseLayer([self.network, self.input_layer], combine_fn = tf.add, name='resnet_add_1')
        self.network = tl.layers.BatchNormLayer(self.network, act = tf.nn.relu, name='resnet_bn_relu_1')
        self.network = tl.layers.MaxPool2d(self.network, name='resnet_maxpool1')

        # Conv fix channel
        self.input_layer = tl.layers.Conv2d(self.network, n_filter=32, act = tf.nn.relu, name='resnet_conv4')

        # -----------------------------
        # 2nd general residual block
        # -----------------------------
        self.network = tl.layers.Conv2d(self.input_layer, n_filter=32, name='resnet_conv5')
        self.network = tl.layers.Conv2d(self.network, n_filter=32, name='resnet_conv6')
        self.network = tl.layers.ElementwiseLayer([self.network, self.input_layer], combine_fn = tf.add, name='resnet_add_2')
        self.network = tl.layers.BatchNormLayer(self.network, act = tf.nn.relu, name='resnet_bn_relu_2')
        self.network = tl.layers.MaxPool2d(self.network, name='resnet_maxpool2')

        # Softmax
        self.network = tl.layers.Conv2d(self.network, n_filter=64, act = tf.nn.relu, name='resnet_conv7')
        self.network = tl.layers.FlattenLayer(self.network, name='resnet_flat')
        self.network = tl.layers.DenseLayer(self.network, n_units = 10, act = tf.nn.softmax, name='resnet_fc')
        self.predict = self.network.outputs
        self.work(imgs_ph, tags_ph, self.predict)

class RiR(Net):
    def __init__(self, imgs_ph, tags_ph):
        self.imgs_ph = imgs_ph
        self.tags_ph = tags_ph

        # 1st conv fix channel
        self.input_layer = tl.layers.InputLayer(self.imgs_ph, name='rir_input')
        self.residual_input = tl.layers.Conv2d(self.input_layer, n_filter=16, act = tf.nn.relu, name='rir_res_conv1')
        self.transient_input = tl.layers.Conv2d(self.input_layer, n_filter=16, act = tf.nn.relu, name='rir_tra_conv1')

        # -----------------------------
        # 1st general residual block
        # -----------------------------
        self.residual_stream = tl.layers.Conv2d(self.residual_input, n_filter=16, name='rir_res_conv2')
        self.residual_stream = tl.layers.Conv2d(self.residual_stream, n_filter=16, name='rir_res_conv3')
        self.transient_stream = tl.layers.Conv2d(self.transient_input, n_filter=16, name='rir_tra_conv2')
        self.transient_stream = tl.layers.Conv2d(self.transient_stream, n_filter=16, name='rir_tra_conv3')
        self.residual_stream = tl.layers.ElementwiseLayer([self.residual_input, self.residual_stream, self.transient_stream], combine_fn = tf.add, name='rir_res_add_1')
        self.transient_stream = tl.layers.ElementwiseLayer([self.residual_stream, self.transient_stream], combine_fn = tf.add, name='rir_tra_add_1')
        self.residual_stream = tl.layers.BatchNormLayer(self.residual_stream, act = tf.nn.relu, name='rir_res_bn_1')
        self.residual_stream = tl.layers.MaxPool2d(self.residual_stream, name='rir_res_pool_1')
        self.transient_stream = tl.layers.BatchNormLayer(self.transient_stream, act = tf.nn.relu, name='rir_tra_bn_1')
        self.transient_stream = tl.layers.MaxPool2d(self.transient_stream, name='rir_tra_pool_1')

        # 2nd conv fix channel
        self.residual_input = tl.layers.Conv2d(self.residual_stream, n_filter=32, act = tf.nn.relu, name='rir_res_conv4')
        self.transient_input = tl.layers.Conv2d(self.transient_stream, n_filter=32, act = tf.nn.relu, name='rir_tra_conv4')

        # -----------------------------
        # 2nd general residual block
        # -----------------------------
        self.residual_stream = tl.layers.Conv2d(self.residual_input, n_filter=32, name='rir_res_conv5')
        self.residual_stream = tl.layers.Conv2d(self.residual_stream, n_filter=32, name='rir_res_conv6')
        self.transient_stream = tl.layers.Conv2d(self.transient_input, n_filter=32, name='rir_tra_conv5')
        self.transient_stream = tl.layers.Conv2d(self.transient_stream, n_filter=32, name='rir_tra_conv6')
        self.residual_stream = tl.layers.ElementwiseLayer([self.residual_input, self.residual_stream, self.transient_stream], combine_fn = tf.add, name='rir_res_add_2')
        self.transient_stream = tl.layers.ElementwiseLayer([self.residual_stream, self.transient_stream], combine_fn = tf.add, name='rir_tra_add_2')
        self.residual_stream = tl.layers.BatchNormLayer(self.residual_stream, act = tf.nn.relu, name='rir_res_bn_2')
        self.residual_stream = tl.layers.MaxPool2d(self.residual_stream, name='rir_res_pool_2')
        self.transient_stream = tl.layers.BatchNormLayer(self.transient_stream, act = tf.nn.relu, name='rir_tra_bn_2')
        self.transient_stream = tl.layers.MaxPool2d(self.transient_stream, name='rir_tra_pool_2')

        # Softmax
        self.network = tl.layers.ConcatLayer([self.residual_stream, self.transient_stream])
        self.network = tl.layers.Conv2d(self.network, n_filter=64, act = tf.nn.relu, name='rir_conv7')
        self.network = tl.layers.FlattenLayer(self.network, name='rir_flat')
        self.network = tl.layers.DenseLayer(self.network, n_units = 10, act = tf.nn.softmax, name='rir_fc')
        self.predict = self.network.outputs
        self.work(imgs_ph, tags_ph, self.predict)