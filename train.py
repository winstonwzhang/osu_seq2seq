
import os
import sys
import time
import tensorflow as tf
import numpy as np
import random
import yaml
import argparse
import pdb

from data_loader import get_dataloaders

# reproducibility
# Set the seed for hash based operations in python
os.environ['PYTHONHASHSEED'] = '0'
random.seed(2020)

# Set the random seed in tensorflow at graph level
tf.random.set_seed(42)



def main():
    # configuration
    configfile = open('config/hparams.yaml')
    config = AttrDict(yaml.load(configfile,Loader=yaml.FullLoader))
    pdb.set_trace()
    
    # get vocab embedding with d_model vectors
    # vocab: list of unique words N
    # word2idx: dict mapping N words to single int index
    # embed: (N,d_model) matrix containing embedding vector for each word
    # sim: (N,N) matrix containing cosine similarity for each word pair i,j
    vocab,word2idx,embed,sim = get_vocab(config.model.d_model,sim_thresh=config.model.sim_thresh)
    config.model.vocab_size = len(vocab)
    
    train_loader, dev_loader, test_loader = get_dataloaders(config, word2idx)
    
    
    
    



if __name__=='__main__':
    main()