
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

#loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
LabelSmoothing_loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
    reduction=tf.losses.Reduction.NONE)

cat_loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
    reduction=tf.losses.Reduction.NONE)


def ClassWeightedWordSimilarityLoss(weights,sim_matrix,vocab_size):
    """
    Classes weighted by frequency of occurrence in training set
    Real labels replaced by soft label cosine similarity scores
    
    pred (FloatTensor): batch_size x sequence length x vocab_size
    real (LongTensor): batch_size x sequence length
    sim_matrix (FloatTensor): vocab_size x vocab size
    weights (FloatTensor): vocab size
    """
    
    def cwwsLoss(real,pred):
        real = tf.cast(real,tf.int32)
        real_onehot = tf.one_hot(real,depth=vocab_size)
        
        # Clip the prediction value to prevent NaN's and Inf's
        #epsilon = K.epsilon()
        #pred = K.clip(pred, epsilon, 1. - epsilon)
        
        # (real label one hot) dot product (similarity matrix)
        # [batch x seq x vocab] dot [vocab x vocab]
        # output shape: [batch x seq x vocab]
        # where each vector along the last [vocab] is no longer one-hot encoded,
        # but instead the cosine similarity soft labels for that specific word class
        real_sim = tf.tensordot(real_onehot,sim_matrix,axes=[[2],[0]])
        unweighted_loss = cat_loss_obj(real_sim, pred)
        
        # deduce weights for batch samples based on their true label
        batch_weights = tf.reduce_sum(weights * real_onehot, axis=-1)
        
        loss_ = unweighted_loss * batch_weights
        pdb.set_trace()
        return tf.reduce_mean(loss_)
    
    return cwwsLoss


def categorical_focal_loss(alpha, gamma=2.,vocab_size=107):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        y_true_oh = tf.one_hot(y_true,depth=vocab_size)
        cross_entropy = -y_true_oh * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


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


def ClassWeightedLabelSmoothingLoss(real,pred,weights,vocab_size,epsilon):
    """
    pred (FloatTensor): batch_size x seq_len x vocab_size
    real (LongTensor): batch_size x seq_len
    weights (FloatTensor): vocab_size
    """
    real = tf.cast(real,tf.int32)
    real_onehot = tf.one_hot(real,depth=vocab_size)
    
    # deduce weights for batch samples based on their true label
    batch_weights = tf.reduce_sum(weights * real_onehot, axis=-1)
    real_smoothed = label_smoothing(real_onehot,epsilon)
    unweighted_loss = LabelSmoothing_loss_object(real_smoothed, pred)
    
    loss_ = unweighted_loss * batch_weights
    #mask = tf.math.logical_not(tf.math.equal(real, 0))
    #mask = tf.cast(mask, dtype=loss_.dtype)

    # Since the target sequences are padded, it is important to apply a padding mask when calculating the loss.
    #loss_ *= mask

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
    real_onehot = tf.one_hot(real,depth=vocab_size)
    
    # (real label one hot) dot product (similarity matrix)
    # [batch x seq x vocab] dot [vocab x vocab]
    # output shape: [batch x seq x vocab]
    # where each vector along the last [vocab] is no longer one-hot encoded,
    # but instead the cosine similarity soft labels for that specific word class
    real_sim = tf.tensordot(real_onehot,sim_matrix,axes=[[2],[0]])
    pdb.set_trace()
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
    
    real = tf.convert_to_tensor([[2,1,0,1],
                                 [0,1,1,2]], tf.int32)
    pred = tf.convert_to_tensor([[[0.1,0.1,0.8],
                                 [0.3,0.6,0.1],
                                 [0.98,0.01,0.01],
                                 [0.1,0.6,0.3]],
                                [[0.1,0.1,0.8],
                                 [0.3,0.6,0.1],
                                 [0.98,0.01,0.01],
                                 [0.4,0.3,0.3]]], tf.float32)
    weights = tf.convert_to_tensor([0.1, 0.8, 0.1], tf.float32)
    sim_matrix = tf.convert_to_tensor([[1.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0]], tf.float32)
    
    
    loss_obj = ClassWeightedWordSimilarityLoss(weights,sim_matrix,3)
    print(loss_obj(real,pred))
    pdb.set_trace()
    
    alpha = weights
    loss_obj = categorical_focal_loss(alpha, gamma=2.,vocab_size=3)
    testout = loss_obj(real,pred)
    
    print(ClassWeightedLabelSmoothingLoss(real,pred,weights,3,0.1))
    
    print(accuracy_function(real,pred))
    print(soft_absdiff(real,pred,sim_matrix))
    pdb.set_trace()
    
    #print(LableSmoothingLoss(real,pred,3,0.1))
    #print(Loss(real,pred))