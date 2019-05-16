
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sys import argv

basedir = argv[1]

im_full = np.array(Image.open(basedir + 'tumor.png'))/255.0
im = np.array(Image.open(basedir + 'tumor_scaled.png'))/255.0
cropped_image = np.array(Image.open(basedir + 'crop.png'))/255.0

width = 50

def middle(im, width):
    n = im.shape[0]//2
    m = width//2

    return im[n-m:n+m, n-m:n+m]

# fig, ax = plt.subplots(2,4)

# Classical sampling
fourier_samples = np.fft.fft2(im)
full_fourier_result = np.abs(np.fft.ifft2(fourier_samples))

Image.fromarray(middle(full_fourier_result*255, 50)).convert('L').save('./cropped/cropped_full.png')
Image.fromarray(np.ones_like(im)*255).convert('L').save('./cropped/samp_full.png')

# Padding
fft_samples = np.fft.fftshift(np.fft.fft2(im_full))

fourier_coeffs = np.zeros((2048, 2048), dtype='complex')
a = 1024-256
fourier_coeffs[a:a+512,a:a+512] = np.fft.fftshift(fourier_samples) #fft_samples[a:a+512,a:a+512]

result2 = np.fft.ifft2(np.fft.fftshift(fourier_coeffs))

result2 = np.abs(result2)
result2 /= result2.max()

# create samp patt image
samp_patt2 = np.zeros_like(fourier_coeffs, dtype=np.float32)
samp_patt2[a:a+512,a:a+512] = 1


Image.fromarray(middle(result2*255, 200)).convert('L').save('./cropped/cropped_padded.png')
Image.fromarray(samp_patt2*255).convert('L').save('./cropped/samp_padded.png')


# ax[0,1].imshow(middle(np.abs(result2), 200), cmap='gray', interpolation=None)
# ax[1,1].imshow(samp_patt2, cmap='gray', interpolation=None)


samp = np.array(Image.open(basedir + 'samp_enhance.png'), dtype=np.bool)
samples = np.where(np.fft.fftshift(samp), np.fft.fft2(im_full), 0)
adjoint = np.abs(np.fft.ifft2(samples))

Image.fromarray(middle(adjoint*255, 200)).convert('L').save('./cropped/cropped_adjoint.png')

# ax[0,2].imshow(middle(adjoint, 200), cmap='gray')
# ax[1,2].imshow(samp, cmap='gray')


# ax[0,3].imshow(cropped_image, cmap='gray')
# ax[1,3].imshow(samp, cmap='gray')



# result = run_pd(im_full, samp, db4, 9, 1, 100)
# image = Image.fromarray(middle(result*255, 200))

# image.show()
# image.save('output.png')



# result3 = np.abs(np.fft.ifft2(np.fft.fftshift(samples)))

# ax[0,2].imshow(middle(result3, 200), cmap='gray', interpolation=None)
# ax[1,2].imshow(samp, cmap='gray', interpolation=None)


# result = run_pd(im_original, np.fft.fftshift(samp), db4, 9, 1, 100)

# ax[0,3].imshow(middle(result, 200), cmap='gray', interpolation=None)
# ax[1,3].imshow(samp, cmap='gray', interpolation=None)

# plt.savefig(basedir[:-1] + '.png')
# plt.show()
