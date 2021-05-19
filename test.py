import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os.path import join
import scipy.misc
import matplotlib
import cv2


def save_image(im, path, name):
    # im = Image.fromarray(im)
    # im.save()
    # scipy.misc.toimage(join(path, name), im)
    matplotlib.image.imsave(join(path, name), im, cmap=plt.cm.gray)
    # cv2.imwrite(join(path, name), im)


def imdisplay(im):
    """
    the function utilize read_image to display an image in a given representation
    :param filename: he filename of an image on disk (could be grayscale or RGB
    :param representation:  representation code, either 1 or 2 defining whether the output should be a grayscale
           image (1) or an RGB image (2).
    :return:
    """
    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()

def warp_flow(img, flow):
    flow = -flow
    h,w = flow.shape[:2]
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img.astype(flow.dtype), flow, None, cv2.INTER_LINEAR)
    return res.astype(img.dtype)

path = "/Users/shaigindin/ex6-noCollection"
name = "bored.jpeg"
black_rec = np.ones([32, 32])
white_rec = np.zeros([32, 32])
prev = np.array(np.vstack([np.hstack([white_rec, black_rec]), np.hstack([black_rec, white_rec])]))
current_image_gray = np.array(np.vstack([np.hstack([white_rec, white_rec]), np.hstack([black_rec, black_rec])]))
# imdisplay(next)
# save_image(bored, path, name)
transorm = cv2.calcOpticalFlowFarneback(prev, current_image_gray, pyr_scale=0.5, levels=3, winsize=30, iterations=3, poly_n=5,
                                        poly_sigma=1.2, flow=None, flags=0)
imdisplay(prev)
imdisplay(current_image_gray)
imdisplay(warp_flow(prev, transorm))
