import time
from datetime import datetime 
from tfwavelets.nodes import dwt2d, idwt2d

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from tfwavelets.dwtcoeffs import db4
from optimization.gpu import build_fista_graph, run_fista


im_filename = '../data/med_images/mri00.npy'
samp_filename = '../data/sampling_patterns/old/med_05.npy'

samp_base = samp_filename.split('/')[-1]
im_base = im_filename.split('/')[-1]

im = np.load(im_filename)
samp_patt = np.fft.fftshift(np.load(samp_filename))

# Parameters
wav_name = 'db4'
if wav_name == 'db4':
    wav = db4

levels = 5
n_iter = 1000

L = 1
lam = 1

result_coeffs = build_fista_graph(im.shape[0], wav, levels)
node = result_coeffs
# real_idwt = idwt2d(tf.real(result_coeffs), wav, levels)
# imag_idwt = idwt2d(tf.imag(result_coeffs), wav, levels)
# node = tf.complex(real_idwt, imag_idwt)

im = np.expand_dims(im, -1)
samp_patt = np.expand_dims(samp_patt, -1)



with tf.Session() as sess:
    start_time = datetime.now().strftime('%F_%T')
    start = time.time()
    result = sess.run(node, feed_dict={'image:0': im,
                                       'sampling_pattern:0': samp_patt,
                                       'L:0': L,
                                       'lambda:0': lam,
                                       'n_iter:0': n_iter})
    end = time.time()
    print(end-start)
    result = np.abs(np.squeeze(result))

    with open('../data/results/data_fista_new.csv', 'a') as outfile:
        outfile.write('{start_time}.png,{im_base},{samp_base},{n_iter},{levels},{wav_name},{lam}\n'.format(
            start_time = start_time,
            im_base=im_base,
            samp_base=samp_base,
            n_iter     = n_iter,
            levels     = levels,
            wav_name   = wav_name,
            lam        = lam))

    imout = Image.fromarray(result)
    imout = imout.convert('L')
    imout.save('fista_new_{start_time}.png'.format(start_time=start_time))

