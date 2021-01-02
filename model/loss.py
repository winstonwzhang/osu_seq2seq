
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

#loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#LableSmoothing_loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

cat_loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
    reduction=tf.losses.Reduction.NONE)


def sim_w_catcrossentropy_loss(real,pred,vocab_size,sim_matrix):
    """
    pred (FloatTensor): batch_size x vocab_size
    real (LongTensor): batch_size
    sim_matrix (numpy matrix): vocab_size x vocab size
    """
    real = tf.cast(real,tf.int32)
    #real_onehot = tf.one_hot(real,depth=vocab_size)
    #orig_loss = cat_loss_obj(real_onehot,pred)
    
    real_sim = tf.convert_to_tensor(sim_matrix[real,:], tf.float32)
    loss_ = cat_loss_obj(real_sim, pred)
    
    return tf.reduce_mean(loss_)


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.
    For example,
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
    outputs = label_smoothing(inputs)
    with tf.Session() as sess:
        print(sess.run([outputs]))
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    Kin = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / Kin)

def LableSmoothingLoss(real,pred,vocab_size,epsilon):
    """
    pred (FloatTensor): batch_size x vocab_size
    real (LongTensor): batch_size
    """
    real = tf.cast(real,tf.int32)
    real_smoothed = label_smoothing(tf.one_hot(real,depth=vocab_size),epsilon)
    loss_ = LableSmoothing_loss_object(real_smoothed, pred)

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)

    # Since the target sequences are padded, it is important to apply a padding mask when calculating the loss.
    loss_ *= mask

    return tf.reduce_mean(loss_)


def Loss(real,pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=-1, output_type=tf.int32))

    #mask = tf.math.logical_not(tf.math.equal(real, 0))
    #accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    #mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.size(accuracies,out_type=tf.float32)#/tf.reduce_sum(mask)


def soft_absdiff(real,pred,sim_matrix):
    '''
    soft label absolute difference in real similarity and predicted score
    the closer to 0 the better the model prediction of soft labels
    real: [batch] vector of ground truth classes
    pred: [batch x vocab size] matrix of batch prediction scores
    sim_matrix: [vocab size x vocab size] matrix of class similarity scores
    '''
    real = tf.cast(real,tf.int32)
    real_sim = tf.convert_to_tensor(sim_matrix[real,:], tf.float32)
    # absolute diff
    absdiff = tf.math.abs(tf.math.subtract(real_sim, pred))
    
    # "true positive-like" word absdiff performance
    mask = tf.math.logical_not(tf.math.equal(real_sim, 0))
    mask = tf.cast(mask,dtype=tf.float32)
    tp_absdiff = absdiff * mask
    tp_absdiff = tf.math.reduce_sum(tp_absdiff) / tf.math.reduce_sum(mask)
    
    # "true negative-like" word absdiff performance
    mask = tf.math.equal(real_sim, 0)
    mask = tf.cast(mask,dtype=tf.float32)
    tn_absdiff = absdiff * mask
    tn_absdiff = tf.math.reduce_sum(tn_absdiff) / tf.math.reduce_sum(mask)
    return tf.math.reduce_mean(absdiff), tp_absdiff, tn_absdiff


if __name__=='__main__':
    import pdb
    tf.compat.v1.enable_eager_execution()
    
    real = tf.convert_to_tensor([2,1,0], tf.int32)
    pred = tf.convert_to_tensor([[0.1,0.1,0.8],
                                 [0.3,0.6,0.1],
                                 [0.98,0.01,0.01]], tf.float32)
    sim_matrix = np.array([[1.0, 0.3, 0.0],
                           [0.3, 1.0, 0.0],
                           [0.5, 0.0, 1.0]])
    print(sim_w_catcrossentropy_loss(real,pred,3,sim_matrix))
    print(accuracy_function(real,pred))
    print(soft_absdiff(real,pred,sim_matrix))
    pdb.set_trace()
    
    #print(LableSmoothingLoss(real,pred,3,0.1))
    #print(Loss(real,pred))