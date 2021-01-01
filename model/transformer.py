
import tensorflow as tf
import numpy as np
from encoder import Encoder
from decoder import Decoder


class Transformer(tf.keras.Model):
    def __init__(self, num_layers=2, d_model=256, num_heads=8, dff=1024, pe_max_len=8000,
                 vocab_size=140, rate=0.1,config=None,logger=None,vocab_embed=None):
        super(Transformer, self).__init__()

        if config is not None:
            num_enc_layers = config.model.encoder_layers
            num_dec_layers = config.model.decoder_layers
            d_model = config.model.d_model
            num_heads = config.model.n_heads
            dff = config.model.d_ff
            pe_max_len = config.model.pe_max_len
            vocab_size = config.model.vocab_size
            rate = config.model.dropout
            
            if logger is not None:
                logger.info('config.model.encoder_layers: '+str(num_enc_layers))
                logger.info('config.model.encoder_layers: '+str(num_dec_layers))
                logger.info('config.model.d_model:   '+str(d_model))
                logger.info('config.model.n_heads:   '+str(num_heads))
                logger.info('config.model.d_ff:      '+str(dff))
                logger.info('config.model.pe_max_len:'+str(pe_max_len))
                logger.info('config.model.vocab_size:'+str(vocab_size))
                logger.info('config.model.dropout:   '+str(rate))
        else:
            print('use default params')
            num_enc_layers = num_layers
            num_dec_layers = num_layers

        self.encoder = Encoder(num_enc_layers, d_model, num_heads, dff,
                                   pe_max_len,'encoder', rate)

        # decoder requires word embed matrix to initialize embedding layer
        self.decoder = Decoder(num_dec_layers, d_model, num_heads, dff,
                               vocab_size, 'decoder',vocab_embed,pe_max_len,rate)

    def call(self, inputs, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):

        inp = tf.cast(inputs[0],tf.float32)
        tar = tf.cast(inputs[1],tf.int32)

        enc_output = self.encoder((inp, enc_padding_mask),training)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            (tar, enc_output,  look_ahead_mask, dec_padding_mask),training)
        
        final_output = dec_output

        return final_output, attention_weights



if __name__=='__main__':
    sample_transformer = Transformer()

    temp_input = tf.random.uniform((32, 32))
    temp_target = tf.random.uniform((32, 32))
    
    fn_out, _ = sample_transformer(inputs=(temp_input, temp_target), training=False,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)

    sample_transformer.summary()

    # tf.keras.utils.plot_model(sample_transformer)
    print(sample_transformer.get_layer('encoder'))
    tp = sample_transformer.trainable_variables
    for i in range(20):
        print(tp[i].name)

    # model = tf.keras.models.Model(inputs=[temp_input,temp_target],outputs=[fn_out])
    # model.summary()
    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

    # summary_writer = tf.keras.callbacks.TensorBoard(log_dir='modules')
    # summary_writer.set_model(model)