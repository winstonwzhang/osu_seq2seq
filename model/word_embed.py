
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def construct_vocab():
    '''
    get all combinations of words
    
    3 [h,slb,sle] x 8 [E,NE,N,NW,W,SW,S,SE] x 4 [c,s,m,f] = 96
    1 [slc] x 8 [E,NE,N,NW,W,SW,S,SE] x 1 [c]             = 8
    3 [spin,b,e]                                          = 3
    96 + 8 + 3 = 107 unique words
    '''
    v1 = ['h','slb','sle']
    v2 = ['E','NE','N','NW','W','SW','S','SE']
    v3 = ['c','s','m','f']
    
    combs = ['e','b','spin']
    for sw1 in v1:
        for sw2 in v2:
            for sw3 in v3:
                combs.append('_'.join([sw1,sw2,sw3]))
    
    for sw2 in v2:
        combs.append('_'.join(['slc',sw2,'c']))
    
    # dict mapping word to index
    word2idx = {k: i for i, k in enumerate(combs)}
    return combs, word2idx


def construct_embed(vocab,d_model=128):
    '''
    create embedding matrix with dimensions (vocab_size,d_model)
    d_model must be >= 15
    '''
    # custom embedding
    d_custom = 15
    Y = np.zeros((len(vocab), d_custom))
    for i,w in enumerate(vocab):
        if '_' in w:
            w1,w2,w3 = w.split('_')
            
            # h,hd,slb,slc,sle
            # ideally h and slb are closer to each other than slc
            if w1 == 'h':     Y[i,0] = 1; Y[i,1] = 0; Y[i,2] = 0; Y[i,3] = 1
            #elif w1 == 'hd':  Y[i,0] = 1; Y[i,1] = 0.5; Y[i,2] = 0; Y[i,3] = 0.5
            elif w1 == 'slb': Y[i,0] = 0.5; Y[i,1] = 0.5; Y[i,2] = 0.5; Y[i,3] = 0.5
            elif w1 == 'sle': Y[i,0] = 0.5; Y[i,1] = 0; Y[i,2] = 1; Y[i,3] = 0
            elif w1 == 'slc': Y[i,4] = 1
            
            # direction
            # ideally we want N to be closer to NE and NW than S
            if w2 == 'N':    Y[i,5] = 1; Y[i,6] = 0; Y[i,7] = 0; Y[i,8] = 0
            elif w2 == 'NE': Y[i,5] = 0.5; Y[i,6] = 0.5; Y[i,7] = 0; Y[i,8] = 0
            elif w2 == 'E':  Y[i,5] = 0; Y[i,6] = 1; Y[i,7] = 0; Y[i,8] = 0
            elif w2 == 'SE': Y[i,5] = 0; Y[i,6] = 0.5; Y[i,7] = 0.5; Y[i,8] = 0
            elif w2 == 'S':  Y[i,5] = 0; Y[i,6] = 0; Y[i,7] = 1; Y[i,8] = 0
            elif w2 == 'SW': Y[i,5] = 0; Y[i,6] = 0; Y[i,7] = 0.5; Y[i,8] = 0.5
            elif w2 == 'W':  Y[i,5] = 0; Y[i,6] = 0; Y[i,7] = 0; Y[i,8] = 1
            elif w2 == 'NW': Y[i,5] = 0.5; Y[i,6] = 0; Y[i,7] = 0; Y[i,8] = 0.5
            
            # speed
            # ideally c closer to s than f
            if w3 == 'c':   Y[i,9] = 1; Y[i,10] = 0; Y[i,11] = 0
            elif w3 == 's': Y[i,9] = 0.5; Y[i,10] = 0.5; Y[i,11] = 0
            elif w3 == 'm': Y[i,9] = 0; Y[i,10] = 1; Y[i,11] = 0.5
            elif w3 == 'f': Y[i,9] = 0; Y[i,10] = 0.5; Y[i,11] = 1

        elif w == 'spin': Y[i,12] = 1
        elif w == 'b': Y[i,13] = 1
        elif w == 'e': Y[i,14] = 1
    
    # extend coding along entire length of [d_model] vector
    reps, rem = np.divmod(d_model, d_custom)
    Y = np.tile(Y,(1,reps))
    # pad right side remaining cols with zeros
    Y = np.pad(Y, ((0,0),(0,rem)), mode='constant', constant_values=0)
    
    # normalize embeddings
    r_sums = Y.sum(axis=1)
    nY = Y / r_sums[:,np.newaxis]
    return nY
    

def get_vocab(d_model, sim_thresh=0.2):
    '''
    Get default vocab embeddings
    d_model: (int) number of dimensions for embed vectors
    sim_thresh: (float) cos similarity threshold btwn 0 and 1
    '''
    vocab, word2idx = construct_vocab()
    embed = construct_embed(vocab,d_model)
    
    # similarity between words for weighted loss calculation
    sim = cosine_similarity(embed)
    # take power of 4 to reduce similarity of far apart words
    sim = sim**5
    # take threshold of closest words for soft ground truth label
    # similar to this paper's approach
    # https://www.aclweb.org/anthology/D18-1525.pdf
    #sim[sim < sim_thresh] = 0
    
    return vocab, word2idx, embed, sim


def visualize_embeddings(vocab, embed):
    '''visualize word representation vectors'''
    # tsne
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # show embedding matrix
    plt.figure(figsize=(12,7))
    plt.imshow(embed)
    plt.show()

    # get 2 dimensional representations
    new_embed = TSNE(n_components=2).fit_transform(embed)
    
    plt.figure(figsize=(20,11))
    plt.scatter(new_embed[:,0],new_embed[:,1],linewidths=10,color='green')
    plt.xlabel("PC1",size=15)
    plt.ylabel("PC2",size=15)
    plt.title("Word Embedding Space",size=20)
    for i, word in enumerate(vocab):
        plt.annotate(word,xy=(new_embed[i,0],new_embed[i,1]))
    plt.show()


def visualize_sim(vocab, sim):
    '''visualize word representation vector similarity'''
    import matplotlib.pyplot as plt
    
    word_idx = 44
    topk = 25
    w_sim = sim[word_idx,:]
    topidx = np.argpartition(w_sim,-topk)[-topk:]
    
    top_w = [vocab[i] for i in topidx]
    top_sim = w_sim[topidx]
    sim_sorted = np.argsort(top_sim)
    top_w = [top_w[i] for i in sim_sorted]
    top_sim = top_sim[sim_sorted]
    
    # bar chart of closest words and their similarity
    plt.figure(figsize=(20,11))
    plt.bar(range(len(top_sim)), top_sim, tick_label=top_w)
    plt.xlabel("Closest Words",size=15)
    plt.ylabel("Cosine Similarity",size=15)
    plt.title("Word Similarities for "+vocab[word_idx],size=20)
    plt.show()


if __name__=='__main__':
    vocab, word2idx, embed, sim = get_vocab(128, sim_thresh=0.2)
    visualize_sim(vocab,sim)
    visualize_embeddings(vocab, embed)
    import pdb; pdb.set_trace()