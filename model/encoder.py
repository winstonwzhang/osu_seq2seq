import tensorflow as tf
import numpy as np
from layers import EncoderLayer
from positional_encoding import positional_encoding


class Encoder(tf.keras.Model):
    '''
    1. Input Embedding
    2. Positional Encoding
    3. N encoder layers
    '''
    
    def __init__(self, num_layers, d_model, num_heads, dff, pe_max_len,name,
                 dp=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        print('self.num_layers(encoder) ',self.num_layers)
        self.rate = dp

        self.pos_encoding = positional_encoding(pe_max_len, self.d_model)

        self.input_proj = tf.keras.models.Sequential(name='en_proj')
        self.input_proj.add(tf.keras.layers.Dense(units=self.d_model,kernel_initializer='glorot_normal'))
        # self.input_proj.add(tf.keras.layers.Dropout(rate=dp))
        #self.input_proj.add(tf.keras.layers.LayerNormalization(epsilon=1e-6))

        #self.dropout = tf.keras.layers.Dropout(rate=0.1, name='en_proj_dp')

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, 'EN'+str(_),dp)
                           for _ in range(num_layers)]

    def call(self, inputs, training):
        
        x = inputs[0] # B*Ti*D
        mask = inputs[1]
        seq_len = tf.shape(x)[1]

        #x = tf.reshape(x,[x.shape[0],x.shape[1],-1])

        # doing projection and adding position encoding.
        x = self.input_proj(x)  # (batch_size, input_seq_len, d_model)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += tf.cast(self.pos_encoding[:, :seq_len, :], x.dtype)

        # print('dropout.rate: ',str(self.dropout.rate))
        # self.dropout.rate = self.rate
        # print('dropout.rate: ', str(self.dropout.rate))
        #x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


if __name__=='__main__':
    
    tf.compat.v1.enable_eager_execution()

    sample_encoder = Encoder(num_layers=2, d_model=256, num_heads=8,dff=1024, pe_max_len=8500,name='Encoder',dp=0.1)

    sample_encoder_output = sample_encoder((tf.random.normal((32,32,512)),None),training=True)

    print(sample_encoder.summary())
    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)