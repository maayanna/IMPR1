##########################################################
# By Maayann Affriat
# username : maayanna
# filename : sol1.py
#########################################################

import numpy as np
from imageio import imread
import skimage.color
import matplotlib.pyplot as plt

NORMALIZE = 255
BINS = 257
MATRIX = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]], dtype=np.float64)

def read_image(filename, representation):
    """
    Function that reads an image file and convert it into a given representation
    :param filename: the filename of an image on disk
    :param representation: representation code, either 1 or 2 defining whether the output should
                           be a grayscale image (1) or an RGB image (2)
    :return: an image represented by a matrix of type np.float64
    """

    color_flag = True #if RGB image
    image = imread(filename)

    float_image = image.astype(np.float64)

    if not np.all(image <= 1):
        float_image /= NORMALIZE #Normalized to range [0,1]

    if len(float_image.shape) != 3 : #Checks if RGB or Grayscale
        color_flag = False

    if color_flag and representation == 1 : #Checks if need RGB to Gray
        return skimage.color.rgb2gray(float_image)

    # Same coloring already
    return float_image


def imdisplay(filename, representation):
    """
    That function displays an image in a given representation
    :param filename: the filename of an image on disk
    :param representation: representation code, either 1 or 2 defining whether the output should
                           be a grayscale image (1) or an RGB image (2)
    :return: None
    """

    image = read_image(filename, representation)
    plt.imshow(image, cmap="gray")
    plt.show()


def rgb2yiq(imRGB):
    """
    This function transforms an RGB image to the YIQ color space
    :param imRGB: RGB image
    :return: YIQ color space
    """
    return np.dot(imRGB, np.array(MATRIX).T)


def yiq2rgb(imYIQ):
    """
    This function transforms an YIQ color space to a RGB image
    :param imYIQ: YIQ color space
    :return: RGB image
    """
    return np.dot(imYIQ, np.linalg.inv(np.array(MATRIX).T))


def histogram_equalize(im_orig):
    """
    This function performs an histogram equalization of a given image
    :param im_orig: the input grayscale or RGB float64 image with values in [0,1]
    :return: A list [im_eq, hist_orig, hist_eq]
    im_eq - equalized image grayscale or RGB float64 image with values in [0,1]
    hist_orig - is a 256 bin histogram of the original image (array with shape(256))
    hist_eq - is a 256 bin histogram of the equalized image (array with shape(256))
    """

    color_flag = False
    image = im_orig


    if len(im_orig.shape) == 3: #RGB image
        color_flag = True
        y_im = rgb2yiq(im_orig)
        image = y_im[:, :, 0]

    image *= NORMALIZE
    hist_orig, bins = np.histogram(image, range(BINS))
    hist_cum = np.cumsum(hist_orig) #cumulative distribution function

    cum = ((hist_cum - hist_cum.min()) / ( hist_cum.max() - hist_cum.min())) * NORMALIZE

    im_eq = cum[image.astype(np.uint8)]

    hist_eq, bins = np.histogram(im_eq, range(BINS)) #before getting back to float64 does the histogram)

    im_eq /= NORMALIZE
    im_eq = im_eq.astype(np.float64)


    if color_flag:
        y_im[:, :, 0] = im_eq
        im_eq = yiq2rgb(y_im)

    im_eq = im_eq.clip(0,1)
    return [im_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    This function performs optimal quantization of a given grayscale or RGB image
    :param im_orig: input grayscale or RGB  to be quantized (float64 image with values in [0,1])
    :param n_quant: number of intensities the output image should have
    :param n_iter: maximum number of iterations of the optimization procedure
    :return: A list[im_quant, eror]
    im_quant - quantized output image
    error - array with shape(n_iter,)(or less) of the total intensities error for each of the quantization procedure
    """
    color_flag = False
    image = im_orig
    error = list()



    if len(im_orig.shape) == 3: #RGB image
        color_flag = True
        y_im = rgb2yiq(im_orig)
        image = y_im[:, :, 0]

    if np.all(image <= 1):
        image *= NORMALIZE
    my_hist, bins = np.histogram(image, 256, (0,255))
    hist_cum = np.cumsum(my_hist)



    z_array = np.array([0]*(n_quant+1)) #init the z_array
    z_array[0] = 0 #minimal value
    z_array[-1] = 255 #maximal value

    q_array = np.zeros(n_quant) #init the q_array
    pixel_per_z = (hist_cum[-1] / n_quant)




    for i in range(1, n_quant): #Getting the z_array (not optimal)
        z_array[i] =np.argwhere(hist_cum>=(pixel_per_z*i)).astype(np.uint8)[0][0]  #first element to be true

    g = np.arange(256)

    for index in range(n_iter):
        z_copy = z_array.copy()

        errors_per_iter = np.zeros(n_quant)

        for i in range(n_quant): #q  calculation
            start = (z_array[i])+1
            end = (z_array[i+1] + 1)
            hist_work = my_hist[start:end]
            g_work = g[start:end]
            sum_up = np.sum(g_work * hist_work) # g*hist
            sum_down =  np.sum(hist_work)
            if sum_down!=0:
                q_array[i] =  sum_up/sum_down
            else:
                q_array[i] = 0

        for i in range(n_quant):  # error calculating after optimisation of z
            start = int(z_array[i])+1
            end = int(z_array[i + 1]) + 1
            err = np.sum(((np.around(q_array[i]) - g[start:end]) ** 2) * my_hist[start:end])
            errors_per_iter[i] = err
        error.append(np.sum(errors_per_iter))

        for i in range(1, n_quant): #First and last element already defined
            z_array[i] = ((q_array[i-1]) + (q_array[i])) / 2 #optimization of the z parts

        if np.array_equal(z_array, z_copy):
            break






    look_up_table = np.array([]) #create look up table
    look_up_table = np.append(look_up_table, [q_array[0]])

    for i in range(1, 1 + n_quant):
        num = q_array[i-1]
        array_use = np.array([num] * int(z_array[i] - z_array[i-1]))
        temp_array = np.append(look_up_table, array_use) #fill the look up table
        look_up_table = temp_array

    look_up_table = np.append(look_up_table, [q_array[-1]])

    im_quant = look_up_table[image.astype(np.uint8)]
    im_quant /= NORMALIZE

    if color_flag:
        y_im[:, :, 0] = im_quant
        im_quant = yiq2rgb(y_im)

    return [im_quant, error]
