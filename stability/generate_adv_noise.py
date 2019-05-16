import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import tensorflow as tf
from tfwavelets.nodes import dwt2d, idwt2d
from tfwavelets.dwtcoeffs import db4
from optimization.gpu.operators import MRIOperator
from optimization.gpu.algorithms import PrimalDual
from optimization.gpu.proximal import BPDNFStar, BPDNG

import time

from generate_images import load_data


def build_graph(tau):
    """Builds graph where the input is an image placeholder + a perturbation variable"""

    # tf_tau = tf.placeholder(tf.float32, shape=(), name='stab_tau')
    tf_lam = tf.placeholder(tf.float32, shape=(), name='stab_lambda')

    N = 128
    wav = db4
    levels = 4
    
    # Build Primal-dual graph
    tf_im = tf.placeholder(tf.complex64, shape=[N,N,1], name='image')
    tf_samp_patt = tf.placeholder(tf.bool, shape=[N,N,1], name='sampling_pattern')

    # perturbation
    tf_rr_real = tf.Variable(tau*tf.random_uniform(tf_im.shape), name='rr_real', trainable=True)
    tf_rr_imag = tf.Variable(tau*tf.random_uniform(tf_im.shape), name='rr_imag', trainable=True)

    tf_rr = tf.complex(tf_rr_real, tf_rr_imag, name='rr')


    tf_input = tf_im + tf_rr

    op = MRIOperator(tf_samp_patt, wav, levels)
    measurements = op.sample(tf_input)

    tf_adjoint_coeffs = op(measurements, adjoint=True)
    adj_real_idwt = idwt2d(tf.real(tf_adjoint_coeffs), wav, levels)
    adj_imag_idwt = idwt2d(tf.imag(tf_adjoint_coeffs), wav, levels)
    tf_adjoint = tf.complex(adj_real_idwt, adj_imag_idwt)

    prox_f_star = BPDNFStar(measurements)
    prox_g = BPDNG()
    alg = PrimalDual(op, prox_f_star, prox_g)

    initial_x = op(measurements, adjoint=True)
    result_coeffs = alg.run(initial_x)

    real_idwt = idwt2d(tf.real(result_coeffs), wav, levels)
    imag_idwt = idwt2d(tf.imag(result_coeffs), wav, levels)
    result_image = tf.complex(real_idwt, imag_idwt)
    # End build primal-dual graph

    # Start building objective function for adv noise

    # the output of PD with no adv. noise
    tf_solution = tf.placeholder(tf.complex64, shape=[N,N,1], name='actual')

    tf_obj = tf.nn.l2_loss(tf.abs(result_image - tf_solution)) - tf_lam * tf.nn.l2_loss(tf.abs(tf_rr))
    # End building objective function for adv noise


    return tf_rr_real, tf_rr_imag, tf_input, result_image, tf_obj, tf_adjoint

num_noise_iter = 100
stab_eta = 0.01
stab_gamma = 0.9
stab_tau = 1e-5
stab_lambda = 0.01
n_iter=1000

eta = 0.1

tf_rr_real, tf_rr_imag, tf_input, tf_recovery, tf_obj, tf_adjoint = build_graph(stab_tau)

mri, _, samp = load_data()


im = np.expand_dims(mri[0], -1)
samp = np.expand_dims(samp, -1)

opt = tf.train.MomentumOptimizer(stab_eta, stab_gamma, use_nesterov=True).minimize(
        -tf_obj, var_list=[tf_rr_real, tf_rr_imag])

norms = [0.0, 2.1610339, 3.50583, 5.178381, 7.3010273]


start = time.time()
with tf.Session() as sess:
    for image_num, (im, max_norm) in enumerate(zip(mri[0:], norms[0:]), 1):

        # if image_num > 1:
        #     exit(1)
        
        im = np.expand_dims(im, -1)
        sess.run(tf.global_variables_initializer())


        # Compute the 'correct' answer. i.e. without adv. noise
        noiseless = sess.run(tf_recovery, feed_dict={'image:0': im,
                                           'sampling_pattern:0': samp,
                                           'sigma:0': 0.5,
                                           'eta:0': eta,
                                           'tau:0': 0.5,
                                           'theta:0': 1,
                                           'n_iter:0': n_iter,
                                           'stab_lambda:0': stab_lambda,
                                           tf_rr_real: np.zeros_like(im),
                                           tf_rr_imag: np.zeros_like(im) })


        for i in range(1,num_noise_iter+1):
            print('{image_num}: {i}/{num} {t}'.format(image_num=image_num, i=i, num=num_noise_iter, t=(time.time()-start)/60))
            
            sess.run(opt, feed_dict={'image:0': im,
                                     'sampling_pattern:0': samp,
                                     'sigma:0': 0.5,
                                     'eta:0': eta,
                                     'tau:0': 0.5,
                                     'theta:0': 1,
                                     'n_iter:0': n_iter,
                                     'stab_lambda:0': stab_lambda,
                                     'actual:0': noiseless})

            rr = sess.run(tf.complex(tf_rr_real, tf_rr_imag))

            adjoint_result = sess.run(tf_adjoint, feed_dict={'image:0': im,
                                               'sigma:0': 0.5,
                                               'eta:0': eta,
                                               'tau:0': 0.5,
                                               'theta:0': 1,
                                               'sampling_pattern:0': samp,
                                               'n_iter:0': n_iter,
                                               'stab_lambda:0': stab_lambda})


            recovery = sess.run(tf_recovery, feed_dict={'image:0': im,
                                                         'sampling_pattern:0': samp,
                                                         'sigma:0': 0.5,
                                                         'eta:0': eta,
                                                         'tau:0': 0.5,
                                                         'theta:0': 1,
                                                         'n_iter:0': n_iter,
                                                         'stab_lambda:0': stab_lambda})

            length = np.linalg.norm(rr)
            print(length)
            

            np.save('./results_pd/im_{image_num}_iter_{iter_num}_adjoint'.format(image_num=image_num, iter_num=i),
                    np.squeeze(adjoint_result))
            np.save('./results_pd/rr_im_{image_num}_iter_{iter_num}'.format(image_num=image_num, iter_num=i),
                    np.squeeze(rr))
            np.save('./results_pd/im_{image_num}_iter_{iter_num}_rec'.format(image_num=image_num, iter_num=i),
                    np.squeeze(recovery))
            np.save('./results_pd/im_{image_num}_iter_{iter_num}_noise'.format(image_num=image_num, iter_num=i),
                    np.squeeze(im+rr))

            # Save the first that is too long, then break.
            if length > max_norm:
                print("MAX REACHED")
                break
