
import os
import sys
import time
import tensorflow as tf
import numpy as np
import random
import yaml
import argparse
import pdb

from model.input_mask import create_masks
from model.model import Osu_transformer
from model.word_embed import get_vocab
from data_loader import get_dataloaders, DataLoader
from util.mylogger import init_logger
from util import AttrDict

# reproducibility
# Set the seed for hash based operations in python
os.environ['PYTHONHASHSEED'] = '0'
random.seed(2020)

# Set the random seed in tensorflow at graph level
tf.compat.v1.set_random_seed(42)



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/hparams.yaml')
    parser.add_argument('-load_model', type=str, default=None)
    parser.add_argument('-model_name', type=str, default='osu_transformer',
                        help='model name')
    parser.add_argument('-log', type=str, default='train.log')
    opt = parser.parse_args()
    
    # configuration
    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile,Loader=yaml.FullLoader))
    
    logger = init_logger()
    log_name = opt.model_name or config.model.name
    log_folder = os.path.join(os.getcwd(),'log',log_name)
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    logger = init_logger(log_folder+'/'+opt.log)
    
    pdb.set_trace()
    
    # get vocab embedding with d_model vectors
    # vocab: list of unique words N
    # word2idx: dict mapping N words to single int index
    # embed: (N,d_model) matrix containing embedding vector for each word
    # sim: (N,N) matrix containing cosine similarity for each word pair i,j
    vocab,word2idx,embed,sim = get_vocab(config.model.d_model,sim_thresh=config.model.sim_thresh)
    config.model.vocab_size = len(vocab)
    
    train_loader, dev_loader, test_loader = get_dataloaders(config, word2idx)
    
    inputs = np.random.randn(32,32,96,22)
    targets = np.random.randint(0,config.model.vocab_size-1,(32,32))
    
    # no need for encoding or decoding mask since spectrogram inputs will never be padded
    # and word inputs will not have padding either
    comb_mask = create_masks(inputs, targets)
    
    OsuT = Osu_transformer(config,logger,embed)
    final_out, attention_weights = OsuT(inputs,targets,True,None,comb_mask,None)
    
    
    
    



if __name__=='__main__':
    main()