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

    def zeroPadding(self, x):
        self.zero_tensor = tf.zeros_like(x)
        return tf.concat([x, self.zero_tensor], axis=3)

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
        self.network = tl.layers.Conv2d(self.input_layer, n_filter=32, name='resnet_conv2')
        self.network = tl.layers.Conv2d(self.network, n_filter=32, name='resnet_conv3')
        padded_input = tl.layers.LambdaLayer(self.input_layer, self.zeroPadding, name='resnet_channel_padding_1')
        self.network = tl.layers.ElementwiseLayer([self.network, padded_input], combine_fn = tf.add, name='resnet_add_1')
        
        self.network = tl.layers.BatchNormLayer(self.network, act = tf.nn.relu, name='resnet_bn_relu_1')
        self.input_layer = tl.layers.MaxPool2d(self.network, name='resnet_maxpool1')        

        # -----------------------------
        # 2nd general residual block
        # -----------------------------
        self.network = tl.layers.Conv2d(self.input_layer, n_filter=64, name='resnet_conv5')
        self.network = tl.layers.Conv2d(self.network, n_filter=64, name='resnet_conv6')
        padded_input = tl.layers.LambdaLayer(self.input_layer, self.zeroPadding, name='resnet_channel_padding_2')
        self.network = tl.layers.ElementwiseLayer([self.network, padded_input], combine_fn = tf.add, name='resnet_add_2')
        self.network = tl.layers.BatchNormLayer(self.network, act = tf.nn.relu, name='resnet_bn_relu_2')
        self.network = tl.layers.MaxPool2d(self.network, name='resnet_maxpool2')

        # Softmax
        self.network = tl.layers.Conv2d(self.network, n_filter=128, act = tf.nn.relu, name='resnet_conv7')
        self.network = tl.layers.FlattenLayer(self.network, name='resnet_flat')
        self.network = tl.layers.DenseLayer(self.network, n_units = 10, act = tf.nn.softmax, name='resnet_fc')
        self.predict = self.network.outputs
        self.work(imgs_ph, tags_ph, self.predict)

class RiR(Net):
    def __init__(self, imgs_ph, tags_ph):
        self.imgs_ph = imgs_ph
        self.tags_ph = tags_ph

        # Revise channel first
        self.residual_input = tl.layers.InputLayer(imgs_ph, name ='rir_residual_input')
        self.transient_input = tl.layers.InputLayer(imgs_ph, name ='rir_transient_input')
        self.residual_stream = tl.layers.Conv2d(self.residual_input, act = tf.nn.relu, n_filter=16, name='rir_revise_conv1')
        self.transient_stream = tl.layers.Conv2d(self.transient_input, act = tf.nn.relu, n_filter=16, name ='rir_revise_conv2')

        # Add general residual blocks
        self.residual_stream, self.transient_stream = self.residual_block(self.residual_stream, self.transient_stream, 32, 1)
        self.residual_stream, self.transient_stream = self.residual_block(self.residual_stream, self.transient_stream, 64, 2)

        # -----------------------------
        # Softmax
        # -----------------------------
        self.network = tl.layers.ConcatLayer([self.residual_stream, self.transient_stream])
        self.network = tl.layers.Conv2d(self.network, n_filter=128, act = tf.nn.relu, name='rir_conv7')

        # Global average pooling
        kernel_height = self.network.outputs.shape[1]
        kernel_width = self.network.outputs.shape[2]
        self.network = tl.layers.MaxPool2d(self.network, filter_size=(kernel_height, kernel_width))
        self.network = tl.layers.FlattenLayer(self.network, name='rir_flat')

        # Softmax
        self.network = tl.layers.DenseLayer(self.network, n_units = 10, act = tf.nn.softmax, name='rir_fc')
        self.predict = self.network.outputs
        self.work(imgs_ph, tags_ph, self.predict)    

    def residual_block(self, residual_input, transient_input, n_filters, name):
        self.residual_stram_main = tl.layers.Conv2d(residual_input, n_filter=n_filters, name='rir_general_residual_block_'+str(name)+'conv1')
        self.residual_stram_extra = tl.layers.Conv2d(residual_input, n_filter=n_filters, name='rir_general_residual_block_'+str(name)+'conv2')
        self.transient_stream_main = tl.layers.Conv2d(residual_input, n_filter=n_filters, name='rir_general_residual_block_'+str(name)+'conv3')
        self.transient_stream_extra = tl.layers.Conv2d(residual_input, n_filter=n_filters, name='rir_general_residual_block_'+str(name)+'conv4')
        self.residual_stram_main = tl.layers.ElementwiseLayer([self.residual_stram_main, self.transient_stream_extra], combine_fn = tf.add, name='rir_general_residual_block_'+str(name)+'add1')
        padded_input = tl.layers.LambdaLayer(residual_input, self.zeroPadding, name='rir_general_residual_block_'+str(name)+'channel_padding')
        self.residual_stram_main = tl.layers.ElementwiseLayer([self.residual_stram_main, padded_input], combine_fn = tf.add, name='rir_general_residual_block_'+str(name)+'add2')
        self.transient_stream_main = tl.layers.ElementwiseLayer([self.transient_stream_main, self.residual_stram_extra], combine_fn = tf.add, name='rir_general_residual_block_'+str(name)+'add3')
        self.residual_stram_main = tl.layers.BatchNormLayer(self.residual_stram_main, act = tf.nn.relu, name='rir_general_residual_block_'+str(name)+'_res_bn')
        self.transient_stream_main = tl.layers.BatchNormLayer(self.transient_stream_main, act = tf.nn.relu, name='rir_general_residual_block_'+str(name)+'_tra_bn')
        return self.residual_stram_main, self.transient_stream_main