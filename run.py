
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from optimization.gpu import build_pd_graph
from tfwavelets.nodes import idwt2d
from tfwavelets.dwtcoeffs import db4
from PIL import Image

from datetime import datetime
import time

im = np.array(Image.open('./tumor.png'))/255.
im_scaled = np.array(Image.open('tumor_scaled.png'))/255.
samp_patt = np.fft.fftshift(np.array(Image.open('./samp_enhance.png'), dtype=np.bool))



def middle(im, width):
    n = im.shape[0]//2
    m = width//2

    return im[n-m:n+m, n-m:n+m]

levels = 9
n_iter = 1000
wav = db4
wav_name= 'db4'
eta = 0.00

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
samp_patt = np.expand_dims(samp_patt, -1)

with tf.Session() as sess:
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

    # with open('../data/results/data.csv', 'a') as outfile:
    #     outfile.write('{start_time}.png,{im_base},{n_iter},{levels},{wav_name},{eta}\n'.format(
    #         start_time = start_time,
    #         im_base=im_base,
    #         samp_base=samp_base,
    #         n_iter     = n_iter,
    #         levels     = levels,
    #         wav_name   = wav_name,
    #         eta        = eta))

    imout = Image.fromarray(255*result)
    imout = imout.convert('L')
    imout.save('../data/results/tumor_{start_time}.png'.format(start_time=start_time))

    imout2 = Image.fromarray(255*middle(result, 200))
    imout2 = imout2.convert('L')
    imout2.save('../data/results/tumor_part_{start_time}.png'.format(start_time=start_time))

