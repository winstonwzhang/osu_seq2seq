
import tensorflow as tf
import numpy as np
from attention import MultiHeadAttention


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff,kernel_initializer='glorot_normal', activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model,kernel_initializer='glorot_normal')  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
    '''
    Each encoder layer consists of
    - Multi-head attention (with padding mask)
    - Point wise feed foward networks
    
    Each of these sublayers has a residual connection around it followed
    by a layer normalization. Residual connections help in avoiding the
    vanishing gradient problem in deep networks.
    
    Output of each sublayer is LayerNorm(x + Sublayer(x)). Normalization is
    done of the d_model (last) axis.
    '''
    
    def __init__(self, d_model, num_heads, dff, name, rate=0.1):
        super(EncoderLayer, self).__init__(name=name)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6,name=name+'_LN1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6,name=name+'_LN2')

        self.dropout1 = tf.keras.layers.Dropout(rate,name=name+'_dp1')
        self.dropout2 = tf.keras.layers.Dropout(rate,name=name+'_dp2')

    def call(self, x, training, mask):
        attn_output, slf_attn_weight = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    '''
    Each decoder layer consists of these sublayers
    - Masked multi-head attention (look ahead and padding masks)
    - Multi-head attention (with padding mask). V (value) and K (key)
      receive encoder outputs as inputs. Q (query) gets output from
      masked multi-head attention sublayer
    - Point wise feed foward networks
    
    Output of each sublayer is LayerNorm(x + Sublayer(x)). Normalization is
    done of the d_model (last) axis. Residual connections are present.
    
    As Q receives the output from decoder's first attention block, and K
    receives the encoder output, the attention weights represent the
    importance given to the decoder's input based on the encoder's output.
    In other words, the decoder predicts the next word by looking at the
    encoder output and self-attending to its own output.
    '''
    
    def __init__(self, d_model, num_heads, dff, name, rate=0.1):
        super(DecoderLayer, self).__init__(name=name)

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6,name=name+'_LN1') # epsilon used to be  1e-6
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6,name=name+'_LN2')
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6,name=name+'_LN3')

        self.dropout1 = tf.keras.layers.Dropout(rate,name=name+'_dp1')
        self.dropout2 = tf.keras.layers.Dropout(rate,name=name+'_dp2')
        self.dropout3 = tf.keras.layers.Dropout(rate,name=name+'_dp3')

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Prenet(tf.keras.layers.Layer):
    '''
    Feature extraction CNN for encoder input spectrograms
    Input spectrograms: (Batch x Freq x Time x Tick)
    Output embeddings:  (Batch x Tick x Embed_dimension)
    '''
    def __init__(self,n=32,k=3,d=512):
        super(Prenet, self).__init__()

        self.c1 = tf.keras.layers.Conv2D(filters=n,
            kernel_size=k,strides=(1,1), padding='same',
            data_format='channels_first',
            kernel_initializer='glorot_normal')
        
        self.c2 = tf.keras.layers.Conv2D(filters=n,
            kernel_size=k,strides=(2,1), padding='same',
            data_format='channels_first',
            kernel_initializer='glorot_normal')
        
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        
        self.maxpl1 = tf.keras.layers.MaxPool2D(
            pool_size=(2,2), strides=(2,2), padding='same',
            data_format='channels_first')
        
        self.flatten = tf.keras.layers.Flatten(data_format='channels_first')
        self.dense1 = tf.keras.layers.Dense(units=d)
        self.d = d

    def call(self,x,training):
        '''
        :param x: B*Ti*F*T
        :return: B*Ti*D
        '''
        x = tf.cast(x,tf.float32)
        # do conv2d on only Freq and Time (F,T) dimensions
        # merge tick dimension with batch dimension to get (B*Ti,1,F,T)
        bdim = x.shape[0]
        tidim = x.shape[1]
        x = tf.reshape(x,(bdim*tidim, 1, x.shape[2], x.shape[3]))
        
        x = self.c1(x)
        x = self.c2(x)
        #x = self.bn1(x,training=training)
        x = self.relu1(x)
        x = self.maxpl1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        
        # now get tick dimension back (B,Ti,D)
        x = tf.reshape(x,(bdim,tidim,self.d))

        return x


if __name__=='__main__':
    
    import pdb
    #tf.compat.v1.enable_eager_execution()
    
    # prenet
    sample_prenet = Prenet()
    print(sample_prenet(tf.random.uniform((32,32,96,8)), False).shape)
    prenet_w = sample_prenet.get_weights()
    for lw in prenet_w:
        print(lw.shape)
    pdb.set_trace()
    # ffn
    sample_ffn = point_wise_feed_forward_network(512, 2048)
    print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)
    
    # encoder
    sample_encoder_layer = EncoderLayer(512, 8, 2048,'encoderlayer')
    sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)
    print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)

    # decoder
    sample_decoder_layer = DecoderLayer(512, 8, 2048,'decoderlayer')
    sample_decoder_layer_output, _, _ = sample_decoder_layer(
        tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
        False, None, None)
    print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)