import numpy as np
import matplotlib.pyplot as plt

from optimization.cpu import run_fista

im = np.load('../data/med_images/mri00.npy')
samp_patt = np.load('../data/sampling_patterns/old/med_05.npy')
# samp_patt = np.ones_like(im, dtype=np.bool)


# Parameters
wav = 'db4'
levels = 5 
L = 2
lam = 10
n_iter = 1000

result, t = run_fista(im, samp_patt, wav, levels, n_iter, lam, L)
print(t)
