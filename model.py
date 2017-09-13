import tensorlayer as tl
import tensorflow as tf

class CNN(object):
    def __init__(self, imgs_ph):
        self.imgs_ph = imgs_ph

    def __call__(self):
        # 1 & 2 conv
        self.network = tl.layers.InputLayer(self.imgs_ph, name='cnn_input')
        self.network = tl.layers.Conv2d(self.network, n_filter=16, name='cnn_conv1')
        self.network = tl.layers.Conv2d(self.network, n_filter=16, name='cnn_conv2')
        self.network = tl.layers.BatchNormLayer(self.network, act = tf.nn.relu, name='cnn_bn_relu_1')
        self.network = tl.layers.MaxPool2d(self.network, name='cnn_maxpool1')

        # 3 & 4 conv
        self.network = tl.layers.Conv2d(self.network, n_filter=32, name='cnn_conv3')
        self.network = tl.layers.Conv2d(self.network, n_filter=32, name='cnn_conv4')
        self.network = tl.layers.BatchNormLayer(self.network, act = tf.nn.relu, name='cnn_bn_relu_2')
        self.network = tl.layers.MaxPool2d(self.network, name='cnn_maxpool2')

        # Softmax
        self.network = tl.layers.FlattenLayer(self.network)
        self.network = tl.layers.DenseLayer(self.network, n_units = 10, act = tf.nn.softmax, name='cnn_fc')
        return self.network.outputs

class ResNet(object):
    def __init__(self, imgs_ph):
        self.imgs_ph = imgs_ph

    def __call__(self):
        # 1 & 2 conv
        self.input_layer_1 = tl.layers.InputLayer(self.imgs_ph, name='resnet_input')
        self.network = tl.layers.Conv2d(self.input_layer_1, n_filter=16, name='resnet_conv1')
        self.network = tl.layers.Conv2d(self.network, n_filter=16, name='resnet_conv2')
        self.network = tl.layers.BatchNormLayer(self.network, act = tf.nn.relu, name='resnet_bn_relu_1')
        self.network = tl.layers.ElementwiseLayer([self.network, self.input_layer_1], combine_fn = tf.add, name ='resnet_add_1')
        self.input_layer_2 = tl.layers.MaxPool2d(self.network, name='resnet_maxpool1')

        # 3 & 4 conv
        self.network = tl.layers.Conv2d(self.input_layer_2, n_filter=32, name='resnet_conv3')
        self.network = tl.layers.Conv2d(self.network, n_filter=32, name='resnet_conv4')
        self.network = tl.layers.BatchNormLayer(self.network, act = tf.nn.relu, name='resnet_bn_relu_2')
        self.network = tl.layers.ElementwiseLayer([self.network, self.input_layer_2], combine_fn = tf.add, name ='resnet_add_2')
        self.network = tl.layers.MaxPool2d(self.network, name='resnet_maxpool2')

        # Softmax
        self.network = tl.layers.FlattenLayer(self.network)
        self.network = tl.layers.DenseLayer(self.network, n_units = 10, act = tf.nn.softmax, name='resnet_fc')
        return self.network.outputs

class RIR(object):
    def __init__(self, imgs_ph):
        self.imgs_ph = imgs_ph

    def __call__(self):
        # 1st general residual block
        self.residual_input_1 = tl.layers.InputLayer(self.imgs_ph, name='rir_input1')
        self.transient_input_1 = tl.layers.InputLayer(self.imgs_ph, name='rir_input2')
        self.residual_stream = tl.layers.Conv2d(self.residual_input_1, n_filter=16, name ='rir_res_conv1')
        self.residual_stream = tl.layers.Conv2d(self.residual_stream, n_filter=16, name ='rir_res_conv2')
        self.transient_stream = tl.layers.Conv2d(self.transient_input_1, n_filter=16, name ='rir_tra_conv1')
        self.transient_stream = tl.layers.Conv2d(self.transient_stream, n_filter=16, name ='rir_tra_conv2')
        self.residual_stream = tl.layers.ElementwiseLayer([self.residual_input_1, self.residual_stream, self.transient_stream], combine_fn = tf.add, name='rir_res_add1')
        self.transient_stream = tl.layers.ElementwiseLayer([self.transient_stream, self.residual_stream], combine_fn = tf.add, name='rir_tra_add1')
        self.residual_stream = tl.layers.BatchNormLayer(self.residual_stream, act = tf.nn.relu, name='rir_res_bn_relu_1')
        self.residual_input_2 = tl.layers.MaxPool2d(self.residual_stream, name='rir_res_maxpool1')
        self.transient_stream = tl.layers.BatchNormLayer(self.transient_stream, act = tf.nn.relu, name='rir_tra_bn_rely_1')
        self.transient_input_2 = tl.layers.MaxPool2d(self.transient_stream, name='rir_tra_maxpool1')

        # 2st general residual block
        self.residual_stream = tl.layers.Conv2d(self.residual_input_2, n_filter=32, name ='rir_res_conv3')
        self.residual_stream = tl.layers.Conv2d(self.residual_stream, n_filter=32, name ='rir_res_conv4')
        self.transient_stream = tl.layers.Conv2d(self.transient_input_2, n_filter=32, name ='rir_tra_conv3')
        self.transient_stream = tl.layers.Conv2d(self.transient_stream, n_filter=32, name ='rir_tra_conv4')
        self.residual_stream = tl.layers.ElementwiseLayer([self.residual_input_2, self.residual_stream, self.transient_stream], combine_fn = tf.add, name='rir_res_add2')
        self.transient_stream = tl.layers.ElementwiseLayer([self.transient_stream, self.residual_stream], combine_fn = tf.add, name='rir_tra_add2')
        self.residual_stream = tl.layers.BatchNormLayer(self.residual_stream, act = tf.nn.relu, name='rir_res_bn_relu_2')
        self.residual_stream = tl.layers.MaxPool2d(self.residual_stream, name='rir_res_maxpool2')
        self.transient_stream = tl.layers.BatchNormLayer(self.transient_stream, act = tf.nn.relu, name='rir_tra_bn_rely_2')
        self.transient_stream = tl.layers.MaxPool2d(self.transient_stream, name='rir_res_maxpool2')

        # Softmax
        self.network = tl.layers.ConcatLayer([self.residual_stream, self.transient_stream])
        self.network = tl.layers.FlattenLayer(self.network)
        self.network = tl.layers.DenseLayer(self.network, n_units = 10, act = tf.nn.softmax, name='rir_fc')
        return self.network.outputs