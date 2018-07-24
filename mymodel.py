import tensorflow as tf
#nkern=[32,32,32,32,32,32,32,32,1]
nkern=[32,32,32,32,32,32,32,32,16,1]
#nkern=[64,32,32,32,32,32,32,32,16,1] #0430 Adam

class globalpath(object):
    def __init__(self,input,state=True,nkern=nkern,inputdim=1,outputdim=1):#state: True if training
        self.Xinput=input
        self.state=state
        self.nkern=nkern
        self.inputdim=inputdim
        self.outputdim=outputdim
        self.net=self.build_main(input)
        self.output=self.net["Deconv4_o"]
        self.w_regulizer=tf.add_n(tf.get_collection("L2"))
    def conv_filter(self, name, kw, kh, n_in, n_out):
        """
        kw, kh - filter width and height
        n_in - number of input channeConv_layerS
        n_out - number of output channeConv_layerS
        """

        kernel_init_val = tf.truncated_normal([kh, kw, n_in, n_out], dtype=tf.float32, stddev=0.1)
        kernel = tf.get_variable(initializer=kernel_init_val, trainable=True, name='w')

        return kernel

    def conv_bias(self, name, n_out):
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.get_variable(initializer=bias_init_val, trainable=True, name='b')

        return biases

    def conv_layer(self, bottom, name, kw, kh, n_out, dw=1, dh=1, bias=False):

        n_in = bottom.get_shape()[-1].value

        with tf.variable_scope(name) as scope:
            filt = self.conv_filter(name, kw, kh, n_in, n_out)
            conv = tf.nn.conv2d(bottom, filt, (1, dh, dw, 1), padding='SAME')
            tf.add_to_collection('L2', tf.contrib.layers.l2_regularizer(1e-8)(filt))
            out = conv
            if bias:
                conv_biases = self.conv_bias(name, n_out)
                bias = tf.nn.bias_add(conv, conv_biases)
                out = bias

            # relu = tf.nn.tanh(bias)
            return out

    def residual_block(self,input,subname=None):
        if subname is not None:
            name='residual_block_'+subname
        else:
            name='residual_block'
        with tf.variable_scope(name):
            #shp=tf.shape(input)#NHWC
            n_in = input.get_shape()[-1].value
            res=self.activation(input,name='First_Act',type='LReLU')
            res=self.conv_layer(res,name='First_conv',kh=3,kw=3,n_out=n_in)
            res=self.activation(res,name='Second_Act',type='LReLU')
            res=self.conv_layer(res,name='Second_conv',kh=3,kw=3,n_out=n_in)
            out=tf.add(res,input,name='Residual_Add')
            return out
    def residual_block_elu(self,input,subname=None):
        if subname is not None:
            name='residual_block_elu'+subname
        else:
            name='residual_block'
        with tf.variable_scope(name):
            #shp=tf.shape(input)#NHWC
            n_in = input.get_shape()[-1].value
            res=self.activation(input,name='First_Act',type='LReLU')
            res=self.conv_layer(res,name='First_conv',kh=3,kw=3,n_out=n_in)
            res=self.activation(res,name='Second_Act',type='LReLU')
            res=self.conv_layer(res,name='Second_conv',kh=3,kw=3,n_out=n_in)
            out=tf.add(res,input,name='Residual_Add')
            return out


    def mpool_op(self, bottom, name, kh=2, kw=2, dh=2, dw=2):
        return tf.nn.max_pool(bottom,
                              ksize=[1, kh, kw, 1],
                              strides=[1, dh, dw, 1],
                              padding='VALID',
                              name=name)

    def upsample(self, X, scale):
        output = tf.keras.layers.UpSampling2D(scale, data_format='channels_last')

        return output(X)

    def pooling_same(self, X, name, scale=(2, 2)):
        # max pooling without changing size. Padding all none maximum value with 0
        with tf.variable_scope('SamePool'):
            pool = self.mpool_op(X, name=name)
            UP = self.upsample(pool, scale)
            S = tf.cast(tf.equal(X, UP), tf.float32)
            S = S * X
            return S

    def deconv(self,bottom, name, kw, kh, n_out, dw=2, dh=2,bias=False):
        with tf.variable_scope(name):

            # filter : [height, width, output_channeConv_layerS, in_channeConv_layerS]

            shape=bottom.get_shape().dims

            w = tf.get_variable('w', [kh, kw, n_out,shape[-1]],
                                initializer=tf.random_normal_initializer(stddev=0.01))
            tf.add_to_collection('L2', tf.contrib.layers.l2_regularizer(1e-8)(w))
            shp=tf.shape(bottom)
            deconv = tf.nn.conv2d_transpose(bottom, filter=w,output_shape=[shp[0],shp[1]*dw,shp[2]*dh,n_out],
                                            strides=[1, dw, dh,1],data_format='NHWC')
            
            if bias:
                conv_biases = self.conv_bias(name, n_out)
                deconv = tf.nn.bias_add( deconv, conv_biases)
                

            return deconv
    def reflat(self,bottom,name,w_out,h_out,n_out):
        with tf.variable_scope(name):
            shp=tf.shape(bottom)
            fc=self.fc_op(bottom,n_out=w_out*h_out*n_out,name='refc')
            newmap=tf.reshape(fc,[shp[0],h_out,w_out,n_out])
            return newmap


    def fc_op(self, input_op, name, n_out,act=None,LN=True):
        n_in = input_op.get_shape()[-1].value

        with tf.variable_scope(name) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([n_in, n_out], dtype=tf.float32, stddev=0.1),
                                     name='w')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[n_out], dtype=tf.float32), name='b')
           # tf.add_to_collection('L2', tf.contrib.layers.l2_regularizer(1e-8)(kernel))

            activation = self.activation(tf.matmul(input_op, kernel) + biases, type=act,BN=LN)
            return activation


    def layernorm(self, name, input):
        with tf.variable_scope(name):
            output = tf.contrib.layers.layer_norm(input, scope='LN')
            return output

    def batchnorm(self,name,input,state):
        with tf.variable_scope(name):

            output=tf.layers.batch_normalization(input,training=state)
            return output

    def activation(self, input, name='activation', type=None,BN=True):
        if BN:
            input = self.batchnorm(name, input,self.state)
        result = {
            'elu': lambda :tf.nn.elu(input,name=name),
            'ReLU': lambda: tf.nn.relu(input, name=name),
            'LReLU': lambda: tf.nn.leaky_relu(input, name=name),
            'tanh': lambda: tf.nn.tanh(input, name=name),
            'sigmoid': lambda: tf.nn.sigmoid(input, name=name)
        }.get(type, lambda: input)()

        # if result == input:
        #     assert (type is not None), print("Undefined activation operator " + type)

        return result

    def build_main(self, input, reuse=False):
        net = {}
        nkern = self.nkern
        conv_large = 5
        conv_small = 1
        conv_mid = 3



        with tf.variable_scope('Global') as scope:
            if reuse:
                scope.reuse_variables()

            with tf.variable_scope('Conv0'):
                net["Conv0"] = self.residual_block( self.conv_layer(input, name="Conv0_C", kh=7, kw=7,dh=1,dw=1, n_out=nkern[0]))#40*40


            with tf.variable_scope('Conv1'):
                net["Conv1"] = self.residual_block_elu(
                    self.conv_layer(net["Conv0"], name="Conv1_C", kh=5, kw=5,dh=2,dw=2,
                                    n_out=nkern[1]))#20*20

            with tf.variable_scope('Conv2'):
                net["Conv2"] = self.residual_block(
                    self.conv_layer(net["Conv1"], name="Conv2_C", kh=3, kw=3,dh=2,dw=2,
                                    n_out=nkern[2]))#10*10

            with tf.variable_scope('Conv3'):
                net["Conv3"] = self.residual_block(
                    self.conv_layer(net["Conv2"], name="Conv3C", kh=3, kw=3,dh=2,dw=2,
                                    n_out=nkern[3]))#5*5
            
            with tf.variable_scope('Conv4'):
                net["Conv4"] = self.residual_block(
                    self.conv_layer(net["Conv3"], name="Conv4C", kh=3, kw=3,dh=2,dw=2,
                                    n_out=nkern[4]))#5*5
            
            with tf.variable_scope('Deconv1'):

                net["Deconv1"] = self.deconv(net["Conv4"], name="Deconv_layer", kh=3, kw=3, n_out=nkern[5])
                net["Deconv1_o"] =self.conv_layer(net["Deconv1"], name="Deconv1_o_C", kh=3, kw=3,n_out=nkern[5])#10*10


            with tf.variable_scope('Deconv2'):
                concat = tf.concat([net["Deconv1_o"], self.residual_block(net["Conv3"])], -1)
                net["Deconv2"] = self.deconv(concat, name="Deconv_layer", kh=3, kw=3, n_out=nkern[6])
                net["Deconv2_o"] =self.conv_layer(net["Deconv2"],name="Deconv2_o_C", kh=3, kw=3,n_out=nkern[6])#20*20

            with tf.variable_scope('Deconv3'):
                concat = tf.concat([net["Deconv2_o"],self.residual_block(net["Conv2"])], -1)

                net["Deconv3"] = self.deconv(concat, name="Deconv_layer", kh=3, kw=3, n_out=nkern[7])
                net["Deconv3_o"] = self.conv_layer(net["Deconv3"], name="Deconv3_o_C", kh=3, kw=3,n_out=nkern[7])#40*40

            with tf.variable_scope('Deconv4'):
                concat = tf.concat([net["Deconv3_o"],self.residual_block(net["Conv1"])], -1)

                net["Deconv4"] = self.deconv(concat, name="Deconv_layer", kh=3, kw=3, n_out=nkern[8],bias=True)
                net["Deconv4_o_pre"] = self.conv_layer(net["Deconv4"], name="Deconv4_o_C_pre", kh=3, kw=3,n_out=nkern[8],bias=False)#40*40
                net["Deconv4_o"] = self.conv_layer(net["Deconv4_o_pre"], name="Deconv4_o_C", kh=3, kw=3,n_out=nkern[9],bias=True)#40*40

        return net

if __name__ == "__main__":
    input=tf.placeholder(tf.float32,[None,80,80,1])
    state=tf.placeholder(tf.bool)#Feed True if training, False if testing
    model=globalpath(input,state)
    output=model.output#same shape as input with channel=1
    with tf.Session() as sess:
        writer=tf.summary.FileWriter('./test',sess.graph)
