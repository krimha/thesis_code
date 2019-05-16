import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from optimization.cpu import run_pd


im = np.load('../data/med_images/mri00.npy')
samp_patt = np.load('../data/sampling_patterns/old/med_05.npy')
# samp_patt = np.ones_like(im, dtype=np.bool)


# Parameters
wav = 'db4'
levels = 5 
n_iter = 1000

sigma = 0.5
tau = 0.5
theta = 1
eta = 10


result, t = run_pd(im, samp_patt, wav, levels, n_iter, eta, sigma, tau, theta)


