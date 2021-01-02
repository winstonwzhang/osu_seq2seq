
import tensorflow as tf
import numpy as np

from transformer import Transformer
from layers import Prenet
from input_mask import create_masks
from word_embed import get_vocab


class Osu_transformer(tf.keras.Model):
    def __init__(self,config,logger=None,vocab_embed=None):
        super(Osu_transformer, self).__init__()
        
        if config is not None:
            n = config.model.prenet_filters
            k = config.model.prenet_kernel
            d = config.model.prenet_dense
        
            if logger is not None:
                logger.info('config.model.prenet_filters: '+str(n))
                logger.info('config.model.prenet_kernel: '+str(k))
                logger.info('config.model.prenet_dense: '+str(d))
            
            self.prenet  = Prenet(n,k,d)
        
        else:
            self.prenet = Prenet()
            print('default prenet n=32,k=3,d=512')
        
        self.transformer = Transformer(config=config,logger=logger,vocab_embed=vocab_embed)

    def call(self,inputs,targets,training,enc_padding_mask,look_ahead_mask,dec_padding_mask):

        out = self.prenet(inputs,training)
        final_out,attention_weights = self.transformer((out,targets),training,
            enc_padding_mask,look_ahead_mask, dec_padding_mask)

        return final_out,attention_weights


if __name__=='__main__':
    
    # if running python model/model.py from parent directory
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)
    
    import yaml
    from util import AttrDict
    from util.mylogger import init_logger
    
    # suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    configfile = open('config\hparams.yaml')
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    print(config.data_name)
    logger = init_logger()
    
    # get vocab embedding with d_model vectors
    # vocab: list of unique words N
    # word2idx: dict mapping N words to single int index
    # embed: (N,d_model) matrix containing embedding vector for each word
    # sim: (N,N) matrix containing cosine similarity for each word pair i,j
    vocab,word2idx,embed,sim = get_vocab(config.model.d_model,sim_thresh=config.model.sim_thresh)
    config.model.vocab_size = len(vocab)
    
    inputs = np.random.randn(32,32,96,22)
    targets = np.random.randint(0,config.model.vocab_size-1,(32,32))
    
    # no need for encoding or decoding mask since spectrogram inputs will never be padded
    # and word inputs will not have padding either
    comb_mask = create_masks(inputs, targets)
    
    OT = Osu_transformer(config,logger,embed)
    final_out, attention_weights = OT(inputs,targets,True,None,comb_mask,None)

    print('final_out.shape:',final_out.shape)
    print('final_out:',final_out)
    import pdb; pdb.set_trace()