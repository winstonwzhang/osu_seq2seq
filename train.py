
import yaml,argparse,os,time
import tensorflow as tf

# reproducibility
# Set the seed for hash based operations in python
os.environ['PYTHONHASHSEED'] = '0'

# Set the random seed in tensorflow at graph level
tf.random.set_seed(42)

