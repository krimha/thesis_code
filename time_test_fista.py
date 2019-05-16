import time
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tfwavelets.dwtcoeffs import db4
from optimization.gpu import build_fista_graph

from sys import argv

# n = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')

gpu = subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8').strip()
host = subprocess.check_output(['hostname']).decode('utf-8').strip()

N = int(argv[1]) # Height and width of image
M = 10 # Number of experiments 
n_iter = 1000 # Number of iterations
p = 0.25
levels = 5

L = 2
lam = 100


im = np.random.random((N,N,1))

# Sample approx p
np.random.seed(0)
samp = np.random.choice([True,False], (N,N,1), True, [p,1-p])

tf_output = build_fista_graph(N, db4, 5)

sess = tf.Session()
outfile = open('../data/time_fista_nobp.csv', 'a')

print('N =',N)
for i in range(1,M+1):
    start_time = time.time()
    result = sess.run(tf_output, feed_dict={'image:0': im,
                                            'sampling_pattern:0': samp,
                                            'L:0': L,
                                            'lambda:0': lam,
                                            'n_iter:0': n_iter})
    time_spent = time.time()-start_time
    print("{}/{} {:0.4f}".format(i,M, time_spent))


    outfile.write('{N},{M},{num_iter},{sub},{levels},{time},{host},{GPU}\n'.format(
        N = N,
        M = M,
        num_iter=n_iter,
        sub = p,
        levels = levels,
        time = time_spent,
        host=host,
        GPU=gpu))




outfile.close()
sess.close()
