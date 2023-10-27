import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.backend import int_shape

# Only look at the main and old model, the others are still WIP
def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output


def get_additive_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return layers.add([input_layer, output])

def conv_block(filters, kernel_size, x, pool_size=2, activation='relu', l2=1e-5, dropout=0.25, res=False):
            if res:
                residual = x
                strides = 1

            y = layers.Conv1D(filters=filters,
                             kernel_size=kernel_size,
                             strides=1,
                             activation=None,
                             use_bias=False,
                             padding='same',
                             kernel_initializer='he_normal',
                             kernel_regularizer=tf.keras.regularizers.l2(l2)
                             )(x)
                
            y = layers.BatchNormalization(momentum=0.9, gamma_initializer="ones")(y)
            y = layers.Activation(activation)(y)
            if res:
                if(filters!=residual.shape[2]):
                    residual = layers.Conv1D(filters=filters, kernel_size=1, strides=1)(residual)
                y = layers.add([y, residual])
            if(pool_size>1 and x.shape[1]>kernel_size):
                y = layers.MaxPool1D(pool_size=pool_size, padding='same')(y)    
            if(dropout>0):
                y = layers.Dropout(dropout)(y)
            return y
        
def dense_block(units, activation, x, dropout=0.5, l2=1e-5):
    x = layers.Dense(units, activation=None, 
                    use_bias=True,  
                    kernel_initializer='he_normal',
                    kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = layers.BatchNormalization(momentum=0.90, gamma_initializer=None)(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout)(x)
    return x

def transformer_layer(x, mha, norm, ff):
    x_norm1 = norm[0](x)
    x_a, w = mha(x_norm1,x_norm1, return_attention_scores = True)
    x_res1 = layers.add([x, x_a])
    x_norm2 = norm[1](x_res1)
    x_ff = get_output(x_res1, ff)
    x_out = layers.add([x_res1, x_ff])
    return x_out, w

def select(num_classes, filter_size=17, num_filters=1024, pool_size=4, num_dense=1024, activation='gelu',
                learningrate=1e-3, seq_shape=(500,4), insertmotif=False, conv_l2=1e-6, dense_l2=1e-3, conv_do=0.15, dense_do=0.5, 
                use_transformer = False, model_type="main", main_model=False, y = None):
    
    input_ = layers.Input(shape=seq_shape)
    if model_type == "main":   
        
        x = input_
        #x = conv_block(num_filters, filter_size, x, pool_size, activation, conv_l2, conv_do, False)
        x = conv_block(num_filters, filter_size, x, pool_size, activation, 1e-5, 0.3, False)

        for i in range(2):
            x = conv_block(int(num_filters/2), 3, x, pool_size, 'relu', conv_l2, conv_do, True)
            #x = conv_block(512, 11, x, pool_size, 'relu', conv_l2, conv_do, False)
        for i in range(1):
            x = conv_block(int(num_filters/2), 3, x, 2, 'relu', conv_l2, conv_do, True)
            x = conv_block(int(num_filters/2), 3, x, 2, 'relu', conv_l2, conv_do, True)
            #x = conv_block(int(num_filters/2), 3, x, 2, 'relu', conv_l2, conv_do, True)
            #x = conv_block(int(num_filters/2), 3, x, 2, 'relu', conv_l2, conv_do, True)
                
        if(use_transformer): 
            numenc = 4
            for i in range(numenc):
                norm_layer_tr = [layers.LayerNormalization(epsilon=1e-6), layers.LayerNormalization(epsilon=1e-6)]
                mha_layer = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=x.shape[-1], dropout=0.3)
                ff_layer = [layers.Conv1D(filters=2*x.shape[-1], kernel_size=1), layers.Dropout(0.4), layers.ReLU(), layers.Conv1D(filters=x.shape[-1], kernel_size=1), layers.Dropout(0.4)]
                x, _ = transformer_layer(x, mha_layer, norm_layer_tr, ff_layer)
        
        #x = layers.Flatten()(x)
        x = layers.GlobalAveragePooling1D()(x)
        #x = layers.Dropout(0.4)(x)
        #x = dense_block(num_dense, 'relu', x, dropout=dense_do, l2=dense_l2)
        
        logits = layers.Dense(num_classes, activation='linear', use_bias=True)(x)

        output =logits
    
    if model_type == "old":
        input_layer = Input(shape=seq_shape)
        conv_layer = [
            layers.Conv1D(filters=1024, kernel_size=24, padding="same", kernel_initializer='glorot_uniform'),
            #layers.BatchNormalization(),
            layers.LeakyReLU()]
        maxp_layer = [
            layers.MaxPool1D(pool_size=16, strides=16, padding='valid'),
            layers.Dropout(0.5)]
        lstm_layer = [
            layers.TimeDistributed(
                layers.Dense(units=256, activation='relu')),
            layers.Bidirectional(
                layers.LSTM(units=256, dropout=0.2, recurrent_dropout=0.2,
                            return_sequences=True)),
                layers.Dropout(0.5)]
        flatten_layer = [
            layers.Flatten()]
        dense_layer = [
            layers.Dense(units=512),
            layers.LeakyReLU(),
            layers.Dropout(0.5)]
        output_layer = [
            layers.Dense(units=num_classes),
            layers.Activation('sigmoid')]
        forward_input = input_layer
        input_ = forward_input
        before_concat_layer = conv_layer + maxp_layer + lstm_layer + flatten_layer + dense_layer
        forward_output_f = get_output(forward_input, before_concat_layer)
        reverse_output_r = get_output(reverse_lambda_ax2(reverse_lambda_ax1(forward_input)), before_concat_layer)
        merged_output = layers.concatenate([forward_output_f, reverse_output_r], axis=1)
        output = get_output(merged_output, output_layer)
        
    elif model_type=='chrombpnet':
        x = input_
        x = conv_block(int(num_filters), filter_size, x, 0, activation, 1e-5, 0.1, False)
        # def conv_block(filters, kernel_size, x, pool_size=2, activation='relu', l2=1e-5, dropout=0.25, res=False):
        n_dil_layers=8
        for i in range(1, n_dil_layers + 1):
        # dilated convolution
            conv_x = layers.Conv1D(num_filters, 
                            kernel_size=3,
                            use_bias=False,
                            padding='valid',
                            activation=None,
                            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                            dilation_rate=2**i)(x)
            conv_x = layers.BatchNormalization(momentum=0.9, gamma_initializer="ones")(conv_x)
            conv_x = layers.Activation('relu')(conv_x)

            x_len = int_shape(x)[1]
            conv_x_len = int_shape(conv_x)[1]
            assert((x_len - conv_x_len) % 2 == 0) # Necessary for symmetric cropping

            
            if(num_filters!=x.shape[2]):
                    x = layers.Conv1D(filters=num_filters, kernel_size=1, strides=1)(x)
            x = layers.Cropping1D((x_len - conv_x_len) // 2)(x)
            x = layers.add([conv_x, x])

            x = layers.Dropout(0.1)(x)
        x = layers.GlobalAveragePooling1D()(x)
        #x = layers.Dropout(0.4)(x)
        #x = dense_block(256, 'relu', x, dropout=0.3, l2=1e-4)
        logits = layers.Dense(num_classes, activation='linear', use_bias=True)(x)

        output =logits

        
    elif model_type == "enf":
        def gelu(x: tf.Tensor) -> tf.Tensor:
            """Applies the Gaussian error linear unit (GELU) activation function.
            Using approximiation in section 2 of the original paper:
            https://arxiv.org/abs/1606.08415
            Args:
            x: Input tensor to apply gelu activation.
            Returns:
            Tensor with gelu activation applied to it.
            """
            return tf.nn.sigmoid(1.702 * x) * x
        conv_layer = [
        layers.Conv1D(filters=num_motifs/2, kernel_size=filter_size, padding="same", kernel_initializer='glorot_uniform', use_bias=False)]
        
        def conv_block_(filters, width = 1):
            return [layers.BatchNormalization(), layers.Activation('gelu'), layers.Conv1D(filters=filters, kernel_size=int(width), padding="same", use_bias=False)]
        print('Input')
        print(input_layer.shape)

        forward_input = input_layer #dnasequence
        input_ = forward_input 

        forward_output_conv1 = get_output(forward_input, conv_layer)
        reverse_output_conv1 = get_output(reverse_lambda_ax2(reverse_lambda_ax1(forward_input)), conv_layer)
        print('Conv 1')
        print(forward_output_conv1.shape)
        conv_block_1 =  conv_block_(num_motifs/2, 1)
        x_fw = get_additive_output(forward_output_conv1, conv_block_1)
        x_rev = get_additive_output(reverse_output_conv1, conv_block_1)
        
        #mp1 = layers.MaxPool1D(pool_size=2 , padding='same')
        #x_fw = mp1(x_fw)
        #x_rev = mp1(x_rev)
        #print('Conv 2')
        #print(forward_output_conv2.shape)
        
        n_tower = 1
        for i in range(n_tower):
            conv_block_i = conv_block_(num_motifs, filter_size/2)
            rconv_block_i = conv_block_(num_motifs, 1)
            x_fw = get_output(x_fw, conv_block_i)
            x_rev = get_output(x_rev, conv_block_i)
            x_fw = get_additive_output(x_fw, rconv_block_i)
            x_rev = get_additive_output(x_rev, rconv_block_i)

        #seq_len = np.floor(500/strides)
        mp = layers.MaxPool1D(pool_size=pool_size, strides=strides, padding='valid')
        #do= layers.Dropout(afterMaxpDO)
        do= layers.Dropout(0)

        forward_output_maxp = do(mp(x_fw))
        reverse_output_maxp = do(mp(x_rev))
        
        #batch_norm = layers.BatchNormalization()
        #x_fw = get_output(x_fw, batch_norm)
        #x_rev = get_output(x_rev, batch_norm)

        x_fw = forward_output_maxp
        x_rev = reverse_output_maxp
        
        x_fw = get_output(x_fw, lstm_layer)
        x_rev = get_output(x_rev, lstm_layer)

        #max_pool1 = MaxPooling1D(pool_size=41, strides=1)(x)
        #max_pool1 = Lambda(lambda s: backend.squeeze(s, axis=1))(max_pool1)
        forward_output_flatten = get_output(x_fw, flatten_layer)
        reverse_output_flatten = get_output(x_rev, flatten_layer)

        forward_output_f = get_output(forward_output_flatten, dense_layer)
        reverse_output_r = get_output(reverse_output_flatten, dense_layer)

        merged_output = layers.concatenate([forward_output_f, reverse_output_r], axis=1)
        output = get_output(merged_output, output_layer)

    elif model_type == "basset":
        # input layer
        input_ = layers.Input(shape=(500,4))
        activation = 'gelu'

        class SoftmaxPooling1D(tf.keras.layers.Layer):
            """Pooling operation with optional weights."""

            def __init__(self, num_features: int = 1,
                       pool_size: int = 2,
                       per_channel: bool = False,
                       w_init_scale: float = 0.0):
                       #name: str = 'softmax_pooling'):
                """Softmax pooling.
                Args:
                  pool_size: Pooling size, same as in Max/AvgPooling.
                  per_channel: If True, the logits/softmax weights will be computed for
                    each channel separately. If False, same weights will be used across all
                    channels.
                  w_init_scale: When 0.0 is equivalent to avg pooling, and when
                    ~2.0 and `per_channel=False` it's equivalent to max pooling.
                  name: Module name.
                """
                super().__init__(self)
                self._pool_size = pool_size
                self._per_channel = per_channel
                self._w_init_scale = w_init_scale
                #self._logit_linear = None
                self._logit_linear = layers.Dense(units = num_features if per_channel else 1, use_bias=False, kernel_initializer=tf.keras.initializers.Identity(w_init_scale) )

            #@snt.once
            def _initialize(self, num_features):
                #self._logit_linear = snt.Linear(output_size=num_features if self._per_channel else 1,with_bias=False,w_init=snt.initializers.Identity(self._w_init_scale))
                self._logit_linear = layers.Dense(units = num_features if self._per_channel else 1, use_bias=False, kernel_initializer=tf.keras.initializers.Identity(self._w_init_scale) )
                
            def call(self, inputs):
                _, length, num_features = inputs.shape
                #self._initialize(num_features)
                inputs = tf.reshape(
                    inputs,
                    (-1, length // self._pool_size, self._pool_size, num_features))
                out = tf.reduce_sum(inputs * tf.nn.softmax(self._logit_linear(inputs), axis=-2), axis=-2)
                print(type(out))
                return out
            def get_config(self):
                config = super().get_config().copy()
                config.update({
                    '_pool_size': self._pool_size,
                    '_per_channel': self._per_channel,
                    '_w_init_scale': self._w_init_scale,
                    '_logit_linear': self._logit_linear,
                })
                return config
                                 
        def conv_block_(filters, kernel_size, x, pool_size=2, activation='relu', l2=1e-5, dropout=0.25, res =False, freeze=False):
            if res:
                residual = x
                strides = 1
            
            if(freeze):
                y_1=layers.Conv1D(filters=248,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 activation=None,
                                 use_bias=False,
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2),
                                 trainable=False
                                 )(x)
                y_2 = layers.Conv1D(filters=264,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 activation=None,
                                 use_bias=False,
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2)
                                 )(x)
                y = layers.Concatenate(axis=2)([y_1,y_2])
            else:
                y = layers.Conv1D(filters=filters,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 activation=None,
                                 use_bias=False,
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2)
                                 )(x)
                
            y = layers.BatchNormalization(momentum=0.9, gamma_initializer="ones")(y)
            y = layers.Activation(activation)(y)
            if res:
                if(filters!=residual.shape[2]):
                    residual = layers.Conv1D(filters=filters, kernel_size=1, strides=1)(residual)
                y = layers.add([y, residual])
            if(pool_size>1 and x.shape[1]>kernel_size):
                #_, _, num_features = x.shape
                #x = SoftmaxPooling1D(num_features, 2 ,True, 2.0)(x)
                y = layers.MaxPool1D(pool_size=pool_size, padding='same')(y)    
            if(dropout>0):
                y = layers.Dropout(dropout)(y)
            return y
        
        
        def dense_block_(units, activation, x, dropout=0.5, l2=1e-5):
            x = layers.Dense(units, activation=None, 
                            use_bias=False,  
                            kernel_initializer='he_normal',
                            kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
            x = layers.BatchNormalization(momentum=0.90, gamma_initializer=None)(x)
            x = layers.Activation(activation)(x)
            x = layers.Dropout(dropout)(x)
            return x
        
        def transformer_layer_(x, mha, norm, ff):
            x_norm1 = norm[0](x)
            x_a, w = mha(x_norm1,x_norm1, return_attention_scores = True)
            x_res1 = layers.add([x, x_a])
            x_norm2 = norm[1](x_res1)
            x_ff = get_output(x_res1, ff)
            x_out = layers.add([x_res1, x_ff])
            return x_out, w
        
        
        x = input_
        small = False
        large = False
        l2 = 1e-6
        convdo = 0.15
        resi = True
        if(small):
            x = conv_block_(512,19,x,4,'gelu',l2,convdo,False, freeze=False)
            x = conv_block_(256,11,x,4,'relu',l2,convdo,False)
            x = conv_block_(256,7,x,4,'relu',l2,convdo,False)
        else:
            x = conv_block_(1024,17,x,4,'gelu',l2,convdo, False)
            
            for i in range(2):
                x = conv_block_(512,11,x,4,'relu',l2,convdo,False)
            for i in range(2):
                x = conv_block_(512,5,x,4,'relu',l2,convdo,True)
                
        tr = False
        if(tr):
            numenc = 4
            for i in range(numenc):
                norm_layer_tr = [layers.LayerNormalization(epsilon=1e-6), layers.LayerNormalization(epsilon=1e-6)]
                mha_layer = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64, value_dim=200, dropout=0.3)
                ff_layer = [layers.Conv1D(filters=400, kernel_size=1), layers.Dropout(0.4), layers.ReLU(), layers.Conv1D(filters=200, kernel_size=1), layers.Dropout(0.4)]
                x, _ = transformer_layer_(x, mha_layer, norm_layer_tr, ff_layer)
        
        x = layers.Flatten()(x)
        #x = layers.Dropout(0.3)(x)
        #print(x.shape)
        x = dense_block_(1024, 'relu', x, dropout=0.5, l2=1e-3) #1e-4
        #print(x.shape)
        #x = dense_block(256, 'relu', x, dropout=0.4, l2=1e-5)
        print(x.shape)
        
        logits = layers.Dense(num_classes, activation='linear', use_bias=True)(x)

        output = layers.Activation('sigmoid')(logits)
    
    elif model_type == "enformer":
        def conv_block_(filters, kernel_size, x, activation='relu', l2=1e-5, dropout=0.25, res =False):
            if res:
                residual = x
                strides = 1
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            x = layers.Conv1D(filters=filters,
                             kernel_size=kernel_size,
                             strides=1,
                             activation=None,
                             use_bias=False,
                             padding='same',
                             kernel_initializer=None,
                             kernel_regularizer=tf.keras.regularizers.l2(l2), 
                             bias_regularizer=None, 
                             activity_regularizer=None,
                             kernel_constraint=None, 
                             bias_constraint=None,
                             )(x)

            if(dropout>0):
                x = layers.Dropout(dropout)(x)
            if res:
                #residual = layers.Conv1D(filters=filters, kernel_size=1, strides=strides)(residual)
                x = layers.add([x, residual])
            
            return x
        
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
            return pos * angle_rates

        def positional_encoding(position, d_model):
            angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                  np.arange(d_model)[np.newaxis, :],
                                  d_model)

            # apply sin to even indices in the array; 2i
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

            # apply cos to odd indices in the array; 2i+1
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

            pos_encoding = angle_rads[np.newaxis, ...]

            return tf.cast(pos_encoding, dtype=tf.float32)

        def point_wise_feed_forward_network(d_model, dff):
            return tf.keras.Sequential([layers.Conv1D(filters=d_model*2, kernel_size=1), layers.Dropout(0.4), layers.ReLU(),layers.Conv1D(filters=d_model, kernel_size=1)])

        class EncoderLayer(layers.Layer):
            def __init__(self,*, d_model, num_heads, dff, rate=0.4):
                super(EncoderLayer, self).__init__()
                val_dim = int(d_model/num_heads)
                self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=64, value_dim=val_dim, dropout=rate)
        
                self.layernorm1 = layers.LayerNormalization(epsilon=1e-5)
                self.layernorm2 = layers.LayerNormalization(epsilon=1e-5)

                self.ffn = point_wise_feed_forward_network(d_model, dff)
                self.dropout1 = tf.keras.layers.Dropout(0)
                self.dropout2 = tf.keras.layers.Dropout(rate)
                self.d_model = d_model

            def call(self, x):

                x = self.layernorm1(x)
                attn_output, weights = self.mha(x, x, return_attention_scores = True)  # (batch_size, input_seq_len, d_model)
                #attn_output = self.dropout1(attn_output, training=training)
                out1 = self.dropout2(x+attn_output)

                x = self.layernorm2(out1)
                x = self.ffn(x)
                x = self.dropout2(x)
                out2 = self.dropout1(x+out1)

                return out2, weights

        class Encoder(layers.Layer):
            def __init__(self,*, num_layers, d_model, num_heads, dff, rate=0.5, seq_len):
                super(Encoder, self).__init__()

                self.norm = layers.LayerNormalization(epsilon=1e-6)
                self.norm2 = layers.LayerNormalization(epsilon=1e-6)
                self.d_model = d_model
                self.seq_len = seq_len
                self.num_layers = num_layers

                self.pos_encoding = positional_encoding(seq_len, self.d_model)

                self.enc_layers = [
                    EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
                    for _ in range(num_layers)]

                #self.dropout = tf.keras.layers.Dropout(rate)

            def call(self, x):

                #seq_len = tf.shape(x)[1]

                #x = self.norm(x)
                # adding embedding and position encoding.
                #x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
                #print(x.shape)
                #x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
                #print(x.shape)
                x += self.pos_encoding[:, :self.seq_len, :]
                #print(x.shape)
                #x = self.mp_dropout(x)
                #x = self.norm2(x)

                for i in range(self.num_layers):
                    x, weights = self.enc_layers[i](x)

                return x, weights  # (batch_size, input_seq_len, d_model)
            
        input_ = layers.Input(shape=(500,4))
        filters=512
        x = input_
        x = layers.Conv1D(filters=filters/2, kernel_size=15, strides=1, activation=None, use_bias=False, padding='same', kernel_initializer=None)(x)
        x = conv_block_(filters=filters/2, kernel_size=1, x=x, activation = 'gelu', l2=1e-5, dropout=0.1, res=True)
        x = layers.MaxPool1D(pool_size=2, padding = 'same')(x)
        
        tower_size=4
        for i in range(tower_size):
            x = conv_block_(filters=filters, kernel_size=5, x=x, activation = 'gelu', l2=1e-5, dropout=0.2, res=False)
            x = conv_block_(filters=filters, kernel_size=1, x=x, activation = 'gelu', l2=1e-5, dropout=0.2, res=True)
            x = layers.MaxPool1D(pool_size=2, padding = 'same')(x)
            
            
        print("=========================================================")
        seq_len=x.shape[1]
        print(seq_len)
        print(x.shape)

        enc = Encoder(num_layers=4, d_model=filters, num_heads=8, dff=filters, rate=0.4, seq_len=seq_len)
        x, _ = enc(x)

        x = conv_block_(filters=filters, kernel_size=1, x=x, activation='gelu',l2=1e-5, dropout=0,res=False)
        x = layers.Dropout(0.05)(x)
        x = layers.Activation('gelu')(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='gelu',kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
        x = layers.Dropout(0.2)
        logits = layers.Dense(num_classes, activation='linear', use_bias=True)(x)

        output = layers.Activation('sigmoid')(logits)
        
        
    elif model_type == "scbasset":
        def conv_block_f(filters, kernel_size, pool_size, x, l2=0, do=0, first = False):
            if (not first):
                x = layers.Activation('gelu')(x)
            x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
            x = layers.BatchNormalization(momentum=0.9, gamma_initializer="ones")(x)
            x = layers.Dropout(do)(x)
            if(pool_size>0):
                x = layers.MaxPool1D(pool_size=pool_size, padding='same')(x)
            return x

        def conv_block_(filters, kernel_size, pool_size, l2=0, do=0, first = False):
            if(first):
                return [ layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(l2)),
                  layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
                  layers.Dropout(do),
                  layers.MaxPool1D(pool_size=pool_size, padding='same')
                  ]
            if(pool_size>0):
                return [
                  layers.Activation('gelu'),
                  layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(l2)),
                  layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
                  layers.Dropout(do),
                  layers.MaxPool1D(pool_size=pool_size, padding='same')
                  ]
            else:
                return [
                  layers.Activation('gelu'),
                  layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(l2)),
                  layers.BatchNormalization(momentum=0.9, gamma_initializer="zeros"),
                  layers.Dropout(do)
                  ]
        def conv_block_res(filters, l2):
            return [layers.Conv1D(filters=filters, kernel_size=1, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(l2))
                  ]
        def dense_block_(units, dropout, seq_len, seq_shape, l2):
            return [layers.Activation('gelu'),
                    #layers.Reshape((1,seq_len * seq_depth)),
                    layers.Flatten(),
                    layers.Dense(units=units, use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(l2)),
                    layers.BatchNormalization(momentum=0.90, gamma_initializer=None),
                    layers.Dropout(rate=dropout)
                   ]

        def _round(x):
            divisible_by = 1
            return int(np.round(x / divisible_by) * divisible_by)

        flatten_layer = [layers.Flatten()]

        rev_path = False
        l2 = 1e-5
        do = 0.2
        
        forward_input = input_layer
        input_ = forward_input
        first_conv_block = conv_block_(288,17,3,l2, do=do, first=True)
        forward_output_f = get_output(forward_input, first_conv_block)
        if(rev_path):
            reverse_output_r = get_output(reverse_lambda_ax2(reverse_lambda_ax1(forward_input)), first_conv_block)

        #first_res_conv_block = conv_block_res(288)
        #forward_output_f = get_additive_output(forward_output_f, first_res_conv_block)
        #reverse_output_r = get_additive_output(reverse_output_r, first_res_conv_block)

        repeat = 5
        current_fw = forward_output_f
        if(rev_path):
            current_rev = reverse_output_r
        rep_filters = 288
        for i in range(repeat):
            #tmp_conv_block = conv_block(_round(rep_filters),5,2,0)
            current_fw = conv_block_f(_round(rep_filters),5,2, current_fw, l2, do=do)
            rep_filters *= 1.122
            #current_fw = get_output(current_fw, tmp_conv_block)
            if(rev_path):
                current_rev = get_output(current_rev, tmp_conv_block)

        second_conv_block = conv_block_(256,1,0, l2, do=do)
        current_fw = get_output(current_fw, second_conv_block)
        if(rev_path):
            current_rev = get_output(current_rev, second_conv_block)

        #second_res_conv_block = conv_block_res(256)
        #current_fw = get_additive_output(current_fw, second_res_conv_block)
        #current_rev = get_additive_output(current_rev, second_res_conv_block)

        _, seq_len, seq_depth = current_fw.shape
        dense_block_1 = dense_block_(32,0.2, seq_len, seq_depth,1e-4)
        current_fw = get_output(current_fw, dense_block_1)
        if(rev_path):
            current_rev = get_output(current_rev, dense_block_1)

        final_gelu = [layers.Activation('gelu')]
        current_fw = get_output(current_fw, final_gelu)
        if(rev_path):
            current_rev = get_output(current_rev, final_gelu)

            final_output = layers.concatenate([current_fw, current_rev], axis=1)
        else:
            final_output = current_fw
        output = get_output(final_output, output_layer)
        
    elif model_type == "bpnet_main":
        forward_input = input_layer
        input_ = forward_input
        forward_output_f = get_output(forward_input, bpnet_conv_layer)
        reverse_output_r = get_output(reverse_lambda_ax2(reverse_lambda_ax1(forward_input)), bpnet_conv_layer)
        for layer in bpnet_dilconv_layer:
            forward_output_f = get_additive_output(forward_output_f, layer)
            reverse_output_r = get_additive_output(reverse_output_r, layer)
        before_concat_layer = bpnet_following_layer
        forward_output_f = get_output(forward_output_f, before_concat_layer)
        reverse_output_r = get_output(reverse_output_r, before_concat_layer)
        merged_output = layers.concatenate([forward_output_f, reverse_output_r], axis=1)
        output = get_output(merged_output, output_layer)

    elif model_type == "hidden":
        forward_input = input_layer
        input_ = forward_input
        before_concat_layer = conv_layer + maxp_layer + lstm_layer + flatten_layer + dense_layer
        forward_output_f = get_output(forward_input, before_concat_layer)
        reverse_output_r = get_output(reverse_lambda_ax2(reverse_lambda_ax1(forward_input)), before_concat_layer)
        merged_output = layers.concatenate([forward_output_f, reverse_output_r], axis=1)
        output = merged_output

    elif model_type == "conv":
        forward_input = input_layer
        input_ = forward_input
        before_concat_layer = conv_layer
        forward_output_f = get_output(forward_input, before_concat_layer)
        reverse_output_r = get_output(reverse_lambda_ax2(reverse_lambda_ax1(forward_input)), before_concat_layer)
        output = [forward_output_f, reverse_output_r]

    elif model_type == "conv_maxp":
        forward_input = input_layer
        input_ = forward_input
        before_concat_layer = conv_layer + maxp_layer
        forward_output_f = get_output(forward_input, before_concat_layer)
        reverse_output_r = get_output(reverse_lambda_ax2(reverse_lambda_ax1(forward_input)), before_concat_layer)
        output = [forward_output_f, reverse_output_r]
        
    elif model_type == "conv_maxp_v2":
        x = input_
        x = conv_block(num_filters, filter_size, x, pool_size, activation, conv_l2, conv_do, False)

        output = x

    elif model_type == "without_conv":
        forward_input = Input(shape=main_model.layers[4].input_shape[1:])
        reverse_input = Input(shape=main_model.layers[4].input_shape[1:])
        input_ = [forward_input, reverse_input]
        before_concat_layer = maxp_layer + lstm_layer + flatten_layer + dense_layer
        forward_output_f = get_output(forward_input, before_concat_layer)
        reverse_output_r = get_output(reverse_input, before_concat_layer)
        merged_output = layers.concatenate([forward_output_f, reverse_output_r], axis=1)
        output = get_output(merged_output, output_layer)

    elif model_type == "without_conv_maxp":
        forward_input = Input(shape=main_model.layers[6].input_shape[1:])
        reverse_input = Input(shape=main_model.layers[6].input_shape[1:])
        input_ = [forward_input, reverse_input]
        before_concat_layer = lstm_layer + flatten_layer + dense_layer
        forward_output_f = get_output(forward_input, before_concat_layer)
        reverse_output_r = get_output(reverse_input, before_concat_layer)
        merged_output = layers.concatenate([forward_output_f, reverse_output_r], axis=1)
        output = get_output(merged_output, output_layer)
        
    elif model_type == "without_conv_maxp_v2":
        print(main_model.layers[1].layers[6])
        print(main_model.layers[1].layers[6].input_shape)
        
        input_ = Input(shape=main_model.layers[1].layers[6].input_shape[1:])
        x = input_

        for i in range(2):
            x = conv_block(int(num_filters/2), 11, x, pool_size, 'relu', conv_l2, conv_do, False)
        for i in range(2):
            x = conv_block(int(num_filters/2), 5, x, pool_size, 'relu', conv_l2, conv_do, True)
                
        if(use_transformer): 
            numenc = 4
            for i in range(numenc):
                norm_layer_tr = [layers.LayerNormalization(epsilon=1e-6), layers.LayerNormalization(epsilon=1e-6)]
                mha_layer = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=x.shape[-1], dropout=0.3)
                ff_layer = [layers.Conv1D(filters=256, kernel_size=1), layers.Dropout(0.4), layers.ReLU(), layers.Conv1D(filters=x.shape[-1], kernel_size=1), layers.Dropout(0.4)]
                x, _ = transformer_layer(x, mha_layer, norm_layer_tr, ff_layer)
        
        x = layers.Flatten()(x)
        x = dense_block(num_dense, 'relu', x, dropout=dense_do, l2=dense_l2)
        
        logits = layers.Dense(num_classes, activation='linear', use_bias=True)(x)

        output = layers.Activation('sigmoid')(logits)

    return input_, output