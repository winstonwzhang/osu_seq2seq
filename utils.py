import re
import numpy as np
import scipy

class lazyval:
    """Decorator to lazily compute and cache a value.
    """
    def __init__(self, fget):
        self._fget = fget
        self._name = None

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self

        value = self._fget(instance)
        vars(instance)[self._name] = value
        return value

    def __set__(self, instance, value):
        vars(instance)[self._name] = value


def parseOsuMp3Filename(name):
    '''Uses regex to find filename info'''
    match = re.search('^[0-9]*\s*(.*)\s+\-\s+(.*)\s+',name)
    artist = match.group(1)
    song = match.group(2)
    if '[' in name:
        # get diff name from enclosed [] brackets, choose closest to end
        diff = re.findall('\[([^\[]*)\]',name)
        if diff:
            diff = diff[-1]
    else:
        diff = None
    return artist,song,diff



def round2Base(x, base=10):
    return base * round(x/base)
    
def roundDown2Base(x, base=10):
    return base * np.floor(x/base)

def roundUp2Base(x, base=10):
    return base * np.ceil(x / base)

def smooth(y, win_size=51, method="sg"):
    '''Smooth signal with moving average'''
    if method == "sg":
        y_smooth = scipy.signal.savgol_filter(y, win_size, 3) # window size, polynomial order
    else:
        box = np.ones(win_size)/win_size
        y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def findNearest(known_arr, test_arr):
    '''
    Find nearest element in known_arr for every element in test_arr
    Return closest indices as well as residuals
    https://stackoverflow.com/questions/20780017/vectorize-finding-closest-value-in-an-array-for-each-element-in-another-array
    '''
    diffs = (test_arr.reshape(1,-1) - known_arr.reshape(-1,1))
    indices = np.abs(diffs).argmin(axis=0)
    residual = np.diagonal(diffs[indices,])
    return indices, residual


def reject_outliers(data, m = 2.):
    '''https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list'''
    d = np.abs(data - np.median(data))
    mdev = np.median(d)  # median absolute distance to median
    s = d/mdev if mdev else 0.
    return data[s<m]


def stepDetect(arr):
    '''Find index of shifts in mean (steps) in array'''
    darr = arr - np.average(arr)
    one_arr = np.ones(int(len(darr)))
    stepwin = np.hstack((one_arr, -1*one_arr))
    darr_step = np.convolve(darr, stepwin, mode='valid')
    # get peaks of convolution
    up = signal.find_peaks(darr_step, width=20)[0]
    dn = signal.find_peaks(-darr_step, width=20)[0]
    
    plt.plot(darr)
    plt.plot(darr_step/10)
    for ii in range(len(up)):
        plt.plot((up[ii], up[ii]), (-1000, 1000), 'r')
    for ii in range(len(dn)):
        plt.plot((dn[ii], dn[ii]), (-1000, 1000), 'g')
    plt.show()
    pdb.set_trace()
    return up, dn


def pattwhere_sequ(pattern, a):
    # find index of all occurrences of subarray pattern in array a
    pattern, a = map(np.asanyarray, (pattern, a))
    k = len(pattern)
    if k>len(a):
        return np.empty([0], int)
    hits = np.flatnonzero(a == pattern[-1])
    for p in pattern[-2::-1]:
        hits -= 1
        hits = hits[a[hits] == p]
    return hits


def consecutive(data, stepsize=1):
    # in: np.array([0,47,48,51,52,53])
    # out: [array([0]), array([47,48]), array([51,52,53])]
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def entropy(arr, base=None):
  """ Computes entropy of array distribution. """
  n_labels = len(arr)
  if n_labels <= 1:
      return 0
  value,counts = np.unique(arr, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1:
      return 0
  ent = 0.
  # Compute entropy
  base = e if base is None else base
  for i in probs:
      ent -= i * log(i, base)

  return ent