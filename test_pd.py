import time
from datetime import datetime 
from tfwavelets.nodes import dwt2d, idwt2d

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from tfwavelets.dwtcoeffs import db4
from optimization.gpu import build_pd_graph, run_pd

from sys import argv

im_filename = '../data/med_images/512/mri500.npy'
# samp_filename = '../data/sampling_patterns/512_pattern_05_low_a.npy'

im_base = im_filename.split('/')[-1]

im = np.load(im_filename)
print(im.max())

# Parameters
wav_name = 'db4'
if wav_name == 'db4':
    wav = db4

levels =7 
n_iter = 1000

# def run(eta):
#     result = run_pd(im, samp_patt, wav, levels, n_iter, eta=eta)
#     with open('../data/results/data.csv', 'a') as outfile:
#         outfile.write('{start_time}.png,{n_iter},{levels},{wav_name},{eta}\n'.format(
#             start_time = start_time,
#             n_iter     = n_iter,
#             levels     = levels,
#             wav_name   = wav_name,
#             eta        = eta))


result_coeffs = build_pd_graph(im.shape[0], wav, levels)
real_idwt = idwt2d(tf.real(result_coeffs), wav, levels)
imag_idwt = idwt2d(tf.imag(result_coeffs), wav, levels)
node = tf.complex(real_idwt, imag_idwt)

im = np.expand_dims(im, -1)

print(argv[1:])

eta = 10
samps = [
        # '512_pattern_05_high_a.npy',
        # '512_pattern_05_mid_a.npy',
        '512_pattern_12_low_a.npy',
        # '512_pattern_05_low_a.npy',
        '512_pattern_12_high_a.npy',
        '512_pattern_12_mid_a.npy',
        ]

eta = 50

with tf.Session() as sess:
    for samp_filename in samps:
        samp_filename = '../data/sampling_patterns/' + samp_filename
        samp_patt = np.fft.fftshift(np.load(samp_filename))
        samp_base = samp_filename.split('/')[-1]
        samp_patt = np.expand_dims(samp_patt, -1)
        start_time = datetime.now().strftime('%F_%T')
        start = time.time()

        result = sess.run(node, feed_dict={'image:0': im,
                                           'sampling_pattern:0': samp_patt,
                                           'sigma:0': 0.5,
                                           'eta:0': eta,
                                           'tau:0': 0.5,
                                           'theta:0': 1.0,
                                           'n_iter:0': n_iter})
        end = time.time()
        print(end-start)
        result = np.abs(np.squeeze(result))

        with open('../data/results/data.csv', 'a') as outfile:
            outfile.write('{start_time}.png,{im_base},{samp_base},{n_iter},{levels},{wav_name},{eta}\n'.format(
                start_time = start_time,
                im_base=im_base,
                samp_base=samp_base,
                n_iter     = n_iter,
                levels     = levels,
                wav_name   = wav_name,
                eta        = eta))

        imout = Image.fromarray(result)
        imout = imout.convert('L')
        imout.save('../data/results/higher_{start_time}_{pattern}.png'.format(start_time=start_time,pattern=samp_filename.split('.')[0]))

