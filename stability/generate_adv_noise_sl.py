import time

import tensorflow as tf
import numpy as np
from PIL import Image

from optimization.gpu.operators import MRIOperator
from optimization.gpu.proximal import SQLassoProx1, SQLassoProx2
from optimization.gpu.algorithms import SquareRootLASSO
from tfwavelets.dwtcoeffs import db4
from tfwavelets.nodes import idwt2d

from generate_images import load_data

def build_graph(tau):
    N = 128
    wav = db4
    levels = 4

    tf_lam = tf.placeholder(tf.float32, shape=(), name='stab_lambda')
    
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

    prox1 = SQLassoProx1() 
    prox2 = SQLassoProx2()

    alg = SquareRootLASSO(op, prox1, prox2, measurements)

    initial_x = op(measurements, adjoint=True)

    result_coeffs = alg.run(initial_x)

    real_idwt = idwt2d(tf.real(result_coeffs), wav, levels)
    imag_idwt = idwt2d(tf.imag(result_coeffs), wav, levels)
    result_image = tf.complex(real_idwt, imag_idwt)

    tf_solution = tf.placeholder(tf.complex64, shape=[N,N,1], name='actual')

    tf_obj = tf.nn.l2_loss(tf.abs(result_image - tf_solution)) - tf_lam * tf.nn.l2_loss(tf.abs(tf_rr))
    # End building objective function for adv noise


    return tf_rr_real, tf_rr_imag, tf_input, result_image, tf_obj, tf_adjoint

    return result_image


num_noise_iter = 200
stab_eta = 0.01
stab_gamma = 0.9
stab_tau = 1e-5
stab_lambda = 0.01

n_iter = 1000
tau = 0.5
sigma = 0.5
lam = 0.001


tf_rr_real, tf_rr_imag, tf_input, tf_recovery, tf_obj, tf_adjoint =  build_graph(stab_tau)

mri, _, samp = load_data()
samp = np.expand_dims(samp, -1)
mri = [np.expand_dims(m, -1) for m in mri]


norms = [0.0, 2.1610339, 3.50583, 5.178381, 7.3010273]

opt = tf.train.MomentumOptimizer(stab_eta, stab_gamma, use_nesterov=True).minimize(
        -tf_obj, var_list=[tf_rr_real, tf_rr_imag])

start = time.time()
with tf.Session() as sess:

    for image_num, (im, max_norm) in enumerate(zip(mri[0:], norms[0:]), 1):

        # if image_num > 1:
        #     exit(1)

        sess.run(tf.global_variables_initializer())

        noiseless = sess.run(tf_recovery, feed_dict={'image:0': im,
                                                      'sampling_pattern:0': samp,
                                                      'sigma:0': sigma,
                                                      'tau:0': tau,
                                                      'lambda:0': lam,
                                                      'n_iter:0': n_iter,
                                                       tf_rr_real: np.zeros_like(im),
                                                       tf_rr_imag: np.zeros_like(im) })



        for i in range(1,num_noise_iter+1):
            print('{image_num}: {i}/{num} {t}'.format(image_num=image_num, i=i, num=num_noise_iter, t=(time.time()-start)/60))
            
            sess.run(opt, feed_dict={'image:0': im,
                                     'sampling_pattern:0': samp,
                                     'sigma:0': sigma,
                                     'tau:0': tau,
                                     'lambda:0': lam,
                                     'n_iter:0': n_iter,
                                     'stab_lambda:0': stab_lambda,
                                     'actual:0': noiseless})

            rr = sess.run(tf.complex(tf_rr_real, tf_rr_imag))

            adjoint_result = sess.run(tf_adjoint, feed_dict={'image:0': im,
                                                             'sampling_pattern:0': samp,
                                                             'sigma:0': sigma,
                                                             'lambda:0': lam,
                                                             'tau:0': tau,
                                                             'n_iter:0': n_iter,
                                                             'stab_lambda:0': stab_lambda,
                                                             'actual:0': noiseless})


            recovery = sess.run(tf_recovery, feed_dict={'image:0': im,
                                                         'sampling_pattern:0': samp,
                                                         'sigma:0': sigma,
                                                         'tau:0': tau,
                                                         'n_iter:0': n_iter,
                                                         'stab_lambda:0': stab_lambda,
                                                         'lambda:0': lam,
                                                         'actual:0': noiseless})
            length = np.linalg.norm(rr)
            print(length)
            

            np.save('./results_sl/im_{image_num}_iter_{iter_num}_adjoint'.format(image_num=image_num, iter_num=i),
                    np.squeeze(adjoint_result))
            np.save('./results_sl/rr_im_{image_num}_iter_{iter_num}'.format(image_num=image_num, iter_num=i),
                    np.squeeze(rr))
            np.save('./results_sl/im_{image_num}_iter_{iter_num}_rec'.format(image_num=image_num, iter_num=i),
                    np.squeeze(recovery))
            np.save('./results_sl/im_{image_num}_iter_{iter_num}_noise'.format(image_num=image_num, iter_num=i),
                    np.squeeze(im+rr))


            # Save the first that is too long, then break.
            if length > max_norm:
                print("MAX REACHED")
                break
