
import os
import sys
import json
import h5py
import numpy as np
import random
from random import shuffle
from itertools import groupby
from operator import itemgetter

random.seed(2020)


def get_dataloaders(config, word2idx):
    '''load train dev and test sets'''
    
    print('getting spectrogram list...')
    try:
        with h5py.File(config.feat_path,'r') as f:
            feat_files = list(f.keys())
    except:
        feat_files = os.listdir(config.feat_path)
    
    # get jsons from spectrogram list
    print('getting json list from spectrograms present...')
    diff_dirs = os.listdir(config.word_path)
    
    # dict with feat file being key, list of jsons being value
    fileset = {}
    chosen_diffs = config.diffs.split(',')
    print('dataset song difficulties: ', chosen_diffs)
    
    for diff_d in diff_dirs:
        if diff_d in chosen_diffs:
            dirpath = config.word_path + os.sep + diff_d
            json_files = os.listdir(dirpath)
            for feat_f in feat_files:
                feat_fpath = config.feat_path + os.sep + feat_f
                feat_basef = os.path.splitext(feat_f)[0]
                jf_matches = [dirpath+os.sep+jf for jf in json_files if feat_basef in jf]
                if feat_fpath in fileset.keys():
                    fileset[feat_fpath].extend(jf_matches)
                else:
                    fileset[feat_fpath] = jf_matches
                # delete feat file if it has no json
                if not fileset[feat_fpath]:
                    fileset.pop(feat_fpath,None)
    
    print('number of feat files with corresponding .json: ',len(fileset))
    
    # divide into train, dev, test
    feat_files = list(fileset.keys())
    num_total = len(feat_files)
    perm_files = random.sample(feat_files,k=num_total)
    
    train_prop = config.train.percent
    dev_prop = config.dev.percent
    test_prop = config.test.percent
    
    num_train = round(train_prop * num_total)
    num_dev = round(dev_prop * num_total)
    
    train_files = perm_files[:num_train]
    dev_files = perm_files[num_train:num_train+num_dev]
    test_files = perm_files[num_train+num_dev:]
    
    print('num train: ',len(train_files))
    print('num dev: ',len(dev_files))
    print('num test: ',len(test_files))
    
    train_fileset = {k: fileset[k] for k in train_files}
    dev_fileset = {k: fileset[k] for k in dev_files}
    test_fileset = {k: fileset[k] for k in test_files}
    
    train_loader = DataLoader(config,'train',train_fileset,word2idx)
    dev_loader = DataLoader(config,'dev',dev_fileset,word2idx)
    test_loader = DataLoader(config,'test',test_fileset,word2idx)
    
    return train_loader, dev_loader, test_loader


class DataLoader():
    '''
    Load spectrogram and json files and create batches for training
    '''

    def __init__(self, config, dataset_type, fileset, word2idx):
        
        self.data_type = dataset_type  # train test dev
        self.word2idx = word2idx  # word to idx dictionary
        self.fileset = fileset  # CQT feat file to .json dictionary
        self.len = len(fileset)
        
        self.sample_size = config.__getattr__(dataset_type).sample_size  # per spec
        self.batch_size = config.__getattr__(dataset_type).batch_size  # batch
        self.shuffle = config.__getattr__(dataset_type).shuffle
        
        self.seq_len = config.seq_length
        
        self.num_oct = config.feature.num_oct
        self.sr = config.feature.sr
        self.extract_ms = config.feature.extract_ms
        
        self.calculate_feat_stats()
    
    
    def calculate_feat_stats(self):
        '''calculate cqt time window to tick conversion stats'''
        num_oct = self.num_oct
        win_size = 2**num_oct
        sr = self.sr
        self.win_ms = (win_size / sr) * 1000
        extract_ms = self.extract_ms  # ms to extract around each tick
        extract_win = round(extract_ms / self.win_ms)
        self.extract_l = round(extract_win // 2)
        self.extract_r = extract_win - self.extract_l
        self.ms2win = lambda x: round(x / self.win_ms / 2)  # quick conversion function

        #print('cqt window time in ms: ', self.win_ms)
        #print('extracted # of cqt windows around each tick: ', extract_win)
        #print('extracted # of cqt windows left of tick: ', self.extract_l)
        #print('extracted # of cqt windows right of tick: ', self.extract_r-1)
        #print('2000 ms corrsponds to cqt window: ',self.ms2win(2000))
    
    
    def get_batch(self):
        
        fileset = self.fileset
        feat_lst = list(fileset.keys())
        shuffle_idx = [i for i in range(len(feat_lst))]
        sample_size = self.sample_size
        batch_size = self.batch_size
        # total batch size will be around sample_size x batch_size
        seq_len = self.seq_len
        ms2win = self.ms2win
        extract_l = self.extract_l
        extract_r = self.extract_r

        result = []
        while 1:
            if self.shuffle == True:
                shuffle(shuffle_idx)
            for i in range(len(feat_lst) // batch_size):
                spec_data = []
                in_tokens = []
                out_tokens = []

                begin = i * batch_size
                end = begin + batch_size
                sub_list = shuffle_idx[begin:end]

                # get data
                for index in sub_list:
                    spec_f = feat_lst[index]
                    spec = np.load(spec_f)

                    # choose random .json file to use
                    json_fs = fileset[spec_f]
                    #json_f = random.choice(json_fs)
                    json_f = json_fs[-1]
                    with open(json_f) as json_fobj:
                        map_data = json.load(json_fobj)

                    # get input tick data and word data
                    ticks = map_data['ticks']
                    words = map_data['words']
                    tick_idx = np.arange(len(words))
                    tokens = np.asarray([self.word2idx[w] for w in words],dtype=np.int32)
                    
                    # get idx mask of all empty, break, and spin sections in song
                    empty_i = [w_i for w_i,w in enumerate(words) if w == 'e' or w == 'b' or w == 'spin']
                    consec_empty = []
                    # only keep long consecutive sections of break and empty
                    for k, g in groupby(enumerate(empty_i), lambda ix: ix[0]-ix[1]):
                        chunk = list(map(itemgetter(1), g))
                        if len(chunk) > 7:
                            consec_empty.extend(chunk)
                    
                    default_mask = np.zeros((len(words),), dtype=np.bool)
                    empty_mask = np.zeros((len(words),), dtype=np.bool)
                    empty_mask[consec_empty] = True

                    # choose [sample_size] random [seq_len] long tick sections
                    # might choose overlapping sections if [seq_len] is long
                    # if section overlaps more than 50% with empty sections, choose another
                    # for seqs chosen, extract spectrogram and word data
                    # spec data in form of [sample_size x seq_len x freq x time]
                    seqs_data = []
                    in_data = []
                    out_data = []
                    ct = 0
                    while len(seqs_data) < sample_size and ct < 30:
                        # get random tick from range [10, word_len-seq_len-10]
                        # prevents out of bounds sequences
                        si = random.randint(10,len(words)-(seq_len+10))
                        mask = np.zeros((len(words),), dtype=np.bool)
                        mask[si:si+seq_len] = True
                        if np.sum(empty_mask & mask) > seq_len/2:
                            ct += 1
                            continue
                        seq = np.where(mask)[0]
                        bad_seq = False
                        if len(seq) == seq_len:
                            ti_specs = []
                            for ti in seq:
                                win_idx = ms2win(ticks[ti])
                                ti_spec = spec[:,win_idx-extract_l:win_idx+extract_r]
                                # empty spectrogram (out of bounds tick index)
                                if ti_spec.shape[1] != extract_l + extract_r:
                                    bad_seq = True
                                ti_specs.append(ti_spec)
                            if bad_seq:
                                ct += 1
                                continue
                            ti_specs = np.stack(ti_specs,axis=0)
                            seqs_data.append(ti_specs)
                            # no BOS token, instead shift tick range to left by 1
                            in_data.append(tokens[seq-1])
                            out_data.append(tokens[seq])
                    
                    spec_data.extend(seqs_data)
                    in_tokens.extend(in_data)
                    out_tokens.extend(out_data)
                
                spec_data = np.stack(spec_data,axis=0)
                in_tokens = np.stack(in_tokens,axis=0)
                out_tokens = np.stack(out_tokens,axis=0)

                # normalize batch spectrograms
                spec_data = (spec_data - spec_data.mean()) / spec_data.std()

                inputs = {
                    'spec_X': spec_data,
                    'dec_X': in_tokens,
                    'dec_Y': out_tokens
                }
                yield inputs
        pass

    def __len__(self):
        return self.len


if __name__=='__main__':
    
    from util import AttrDict
    from model.word_embed import get_vocab
    import yaml
    import pdb
    # configuration
    configfile = open('config/hparams.yaml')
    config = AttrDict(yaml.load(configfile,Loader=yaml.FullLoader))
    
    vocab,word2idx,embed,sim = get_vocab(config.model.d_model,sim_thresh=config.model.sim_thresh)
    config.model.vocab_size = len(vocab)
    
    train_loader, dev_loader, test_loader = get_dataloaders(config, word2idx)
    pdb.set_trace()