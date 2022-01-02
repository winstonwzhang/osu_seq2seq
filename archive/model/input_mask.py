import tensorflow as tf
import numpy as np


def create_padding_mask(seq):
    '''
    :param seq: [batch_size * seq_len_k] # k means key in MultiheadAttention
    :return: [batch_size, 1, 1, seq_len_k]
    '''
    if seq.dtype != np.int32:
        #print("float")
        seq = tf.cast(tf.math.equal(seq, 0.), tf.float32)
    else:
        #print("int")
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis,tf.newaxis, :]  # (batch_size, 1,1, seq_len)


def create_look_ahead_mask(size):
    '''
    :param size: == seq_len_k
    :return: (seq_len_q, seq_len_k)
    
    The look-ahead mask is used to mask the future tokens in a sequence.
    In other words, the mask indicates which entries should not be used.

    This means that to predict the third word, only the first and second
    word will be used. Similarly to predict the fourth word, only the first,
    second and the third word will be used and so on.
    '''
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    '''
    :param inp: [batch_size * seq_len_k_of_encoder ]
    :param tar: [batch_size * seq_len_q_of_decoder_block1 ]
    :return:
    '''
    # Encoder padding mask
    #enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    # encoder outputs [batch_size * seq_len * d_model] 中间那一维相比原始encoder的input不变，所以就按照inp计算了
    #dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    #combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    combined_mask = look_ahead_mask

    # print('enc_padding_mask',enc_padding_mask)
    # print('combined_mask', combined_mask)
    # print('dec_padding_mask', dec_padding_mask)

    return combined_mask


def create_DecBlock1_pad_mask(tar):
    tar = tf.cast(tf.math.equal(tar, PAD), tf.float32)
    return tar[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1,1, seq_len)


def create_combined_mask(tar):
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_DecBlock1_pad_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return combined_mask


if __name__=='__main__':
    tf.compat.v1.enable_eager_execution()
    x = tf.constant([[7, 6, 0, 0, 1],
                     [1, 2, 3, 0, 0],
                     [0, 0, 0, 4, 5]])
    print(create_padding_mask(x))
    
    x = tf.random.uniform((1, 3))
    temp = create_look_ahead_mask(x.shape[1])
    print(temp)