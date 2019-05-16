import numpy as np
import matplotlib.pyplot as plt

from optimization.cpu import run_fista
import subprocess

# im = np.load('../data/med_images/mri00.npy')
# samp_patt = np.load('../data/sampling_patterns/old/med_05.npy')
# samp_patt = np.ones_like(im, dtype=np.bool)

host = subprocess.check_output(['hostname']).decode('utf-8').strip()

# Parameters
wav = 'db4'
levels = 5 
n_iter = 1000

L = 2
lam = 10

p = 0.25

for N in (256,512,1024,2048):
    print('N=', N)
    im = np.random.random((N,N))
    samp_patt = np.random.choice([True,False], (N,N), True, [p,1-p])
    for i in range(1,11):
        with open('../data/time_np_fista.csv', 'a') as outfile:
            result, t = run_fista(im, samp_patt, wav, levels, n_iter, lam, L)
            print('{i}/10: {t}'.format(i=i, t=t))

            outfile.write('{host},{N},{t}\n'.format(host=host,N=N,t=t))


