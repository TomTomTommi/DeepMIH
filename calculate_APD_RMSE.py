'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import math
import numpy as np
import cv2
import glob
from natsort import natsorted

def main():
    # Configurations
    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    folder_GT = '/home/jjp/DeepMIH/image/cover/'
    folder_Gen = '/home/jjp/DeepMIH/image/steg_1/'
    crop_border = 1
    suffix = '_secret_rev'  # suffix for Gen images

    APD_all = []
    RMSE_all = []
    img_list = sorted(glob.glob(folder_GT + '/*'))
    img_list = natsorted(img_list)

    for i, img_path in enumerate(img_list):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # base_name = base_name[:5]
        im_GT = cv2.imread(img_path) / 255.
        # print(base_name)
        # print(img_path)
        # print(os.path.join(folder_Gen, base_name + '.png'))
        im_Gen = cv2.imread(os.path.join(folder_Gen, base_name + '.png')) / 255.

        im_GT_in = im_GT
        im_Gen_in = im_Gen

        # # crop borders
        # if im_GT_in.ndim == 3:
        #     cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
        #     cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
        # elif im_GT_in.ndim == 2:
        #     cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
        #     cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
        # else:
        #     raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

        # calculate PSNR and SSIM
        APD = calculate_apd(im_GT_in * 255, im_Gen_in * 255)
        RMSE = calculate_rmse(im_GT_in * 255, im_Gen_in * 255)

        print('{:3d} - {:25}. \tAPD: {:.6f} , \tRMSE: {:.6f}'.format(
            i + 1, base_name, APD, RMSE))
        APD_all.append(APD)
        RMSE_all.append(RMSE)
    print('Average: APD: {:.6f} , RMSE: {:.6f}'.format(
        sum(APD_all) / len(APD_all),
        sum(RMSE_all) / len(RMSE_all)))


def calculate_rmse(img1, img2):
    """
    Root Mean Squared Error
    Calculated individually for all bands, then averaged
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    rmse = np.sqrt(mse)

    return np.mean(rmse)


def calculate_apd(img1, img2):

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    apd = np.mean(np.abs(img1 - img2))
    if apd == 0:
        return float('inf')

    return np.mean(apd)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


if __name__ == '__main__':
    main()
