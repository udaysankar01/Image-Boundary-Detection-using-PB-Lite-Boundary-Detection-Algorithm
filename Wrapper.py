#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import math
from sklearn.cluster import KMeans
import os

def get_2D_gaussian_kernel(kernel_size, sigma, elongation_factor = 1):
    
    range = int((kernel_size - 1)/2)
    range_spread = np.linspace(-range, range, kernel_size)
    x_grid, y_grid = np.meshgrid(range_spread, range_spread)
    
    sigma_x = sigma
    sigma_y = sigma * elongation_factor

    kernel = (1 / np.sqrt(2 * np.pi * sigma_x * sigma_y)) * np.exp(-0.5 * (x_grid.ravel() ** 2 / sigma_x ** 2 + y_grid.ravel() ** 2 / sigma_y ** 2))
    kernel_2D = np.reshape(kernel, (kernel_size, kernel_size))
    
    return kernel_2D

def oriented_DoG(sigma, orientation):

    # selecting the kernel size as 7
    kernel_size = 7

    # establishing the sobel and gaussian kernels for getting the oriented Derivative of Gaussian kernels
    sobel_x = np.array([[+1, 0, -1], [+2, 0, -2], [+1, 0, -1]])
    gaussian_kernel = get_2D_gaussian_kernel(kernel_size, sigma)

    # convolving the sobel filter on the gaussian kernel to get the DoG filter
    dog_filter = cv2.filter2D(gaussian_kernel, ddepth = -1, kernel = sobel_x)

    # using rotation matrix to set the orientation of the DoG filter
    center = (int(kernel_size / 2), int(kernel_size / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center = center, angle = orientation, scale = 1)
    oriented_dog_filter = cv2.warpAffine(src = dog_filter, M = rotation_matrix, dsize = (kernel_size, kernel_size))

    return oriented_dog_filter


def get_and_plot_oriented_DoG (get_filters = False, plot_filters = False, save_figure = False):

    # the oriented derivative of gaussian functions will be appended into an empty list
    DoG = []

    # setting the scales, orientations and sigma values
    scales = [1, np.sqrt(2)]
    s = len(scales)
    o = 16 # number of orientations

    orientations = np.arange(0 , 360, 360/o)

    # defining a subplots for oriented DoG filters (2x16) if plot_filters parameter is True
    if plot_filters:
        fig, axs = plt.subplots(len(scales), o, figsize = (11,5))

    for i, sigma in enumerate(scales):
        if plot_filters:
            axs_i = axs[i] if s > 1 else axs

        for j, orientation in enumerate(orientations):
            if plot_filters:
                axs_j = axs_i[j] if o > 1  else axs_i

            filter = oriented_DoG(sigma, orientation)
            DoG.append(filter)
            # the filter values need to be normalized to the range of 0 to 255 before being showed in the subplot
            filter_normalized = cv2.normalize(filter, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            if plot_filters:
                axs_j.imshow(filter_normalized, cmap='gray')
                axs_j.axis('off')
    if plot_filters:
        if save_figure:
            plt.savefig('./Code/Results/Filter Banks/DoG.png', dpi=500, bbox_inches="tight")
        plt.show()
    if get_filters:
        return DoG

def gaussian_derivative_1D(sigma, order, x, kernel_size = 49):

    variance = sigma ** 2
    gaussian = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * (x ** 2 / variance))

    if order == 0:
        return gaussian

    # first order derivative of gaussian
    elif order == 1:
        return gaussian * (-x / variance)

    # second order derivative of gaussian
    elif order == 2:
        return gaussian * ((x * x) -variance) / (variance ** 2)


def gaussian_derivative_2D(sigma, sigma_scale, order_x, order_y, orientation, kernel_size = 49):
    
    variance = sigma ** 2
    range = int((kernel_size - 1)/2)
    range_spread = np.linspace(-range, range, kernel_size)
    x_grid, y_grid = np.meshgrid(range_spread, range_spread)

    kernel_x = gaussian_derivative_1D(sigma_scale * sigma, order_x, x_grid.ravel(), kernel_size)
    kernel_y = gaussian_derivative_1D(sigma, order_y, y_grid.ravel(), kernel_size)
    kernel = np.reshape(kernel_x * kernel_y, (kernel_size, kernel_size))

    # using rotation matrix to set the orientation of the DoG filter
    center = (int(kernel_size / 2), int(kernel_size / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center = center, angle = orientation, scale = 1)
    oriented_kernel = cv2.warpAffine(src = kernel, M = rotation_matrix, dsize = (kernel_size, kernel_size))

    return oriented_kernel


def LoG(sigma, kernel_size = 49):
    
    variance = sigma ** 2
    range = int((kernel_size - 1) / 2)
    y = np.linspace(-range, range, kernel_size).astype(int)
    x = y.reshape((kernel_size, 1))

    # g = get_2D_gaussian_kernel(kernel_size, sigma)
    g = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-(x*x + y*y) / (2 * variance))
    h = g * ((x*x + y*y) - variance)/(variance**2)
    return h
 

def LM_filters(scales, kernel_size = 49):
    
    o = 6 # number of orientations
    orientations = -np.arange(0 , 180, 180/o)

    # for first and second order derivatives of Gaussian
    LM_filter_bank = []

    for sigma in scales[:3]:
        scale_filters = []

        # first order derivative of gaussian filters
        for orientation in orientations:
            scale_filters.append(gaussian_derivative_2D(sigma, 3, 0, 1, orientation, kernel_size))

        # second order derivative of gaussian filters
        for orientation in orientations:
            scale_filters.append(gaussian_derivative_2D(sigma, 3, 0, 2, orientation, kernel_size))
        LM_filter_bank.append(scale_filters)
    

    # for the Laplacian of Gaussian and Gaussian filters
    log_g_filters = []

    # Laplacian of Gaussian filters
    log_scale_factor = 3
    log_scales = np.array(([scales] + [log_scale_factor * scales])).ravel()
    for sigma in log_scales:
        log_g_filters.append(LoG(sigma, kernel_size))
    
    # Gaussian filters
    for sigma in scales:
        # log_g_filters.append(get_2D_gaussian_kernel(kernel_size, sigma))
        log_g_filters.append(gaussian_derivative_2D(sigma, 1, 0, 0, 0, kernel_size))
    
    LM_filter_bank.append(log_g_filters)

    return LM_filter_bank


def get_and_plot_LM (get_filters = False, plot_filters = False, save_figure = False):

    """
    Leung-Malik Filters
    Reference: https://www.robots.ox.ac.uk/~vgg/research/texclass/code/makeLMfilters.m
    """

    kernel_size = 49
    scales = np.array([1, np.sqrt(2), 2, 2 * np.sqrt(2)])
    LM_return_filter = []

    # LM Small filters
    if get_filters == 'small':
        LMS_filter_bank = LM_filters(scales, kernel_size)
        if plot_filters:
            fig, axs = plt.subplots(np.shape(LMS_filter_bank)[0],np.shape(LMS_filter_bank)[1], figsize = (11,5))
            for i, filter_group in enumerate(LMS_filter_bank):
                for j, filter in enumerate(filter_group):
                    filter_normalized  = cv2.normalize(filter,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                    axs[i, j].imshow(filter_normalized, cmap='gray')
                    axs[i, j].axis('off')
            if save_figure:
                plt.savefig('./Code/Results/Filter Banks/LMS.png', dpi = 500, bbox_inches="tight")
            plt.show()

        for filter_group in LMS_filter_bank:
            for filter in filter_group:
                LM_return_filter.append(filter)
        return LM_return_filter
    
    # LM Large Filters
    if get_filters == 'large':
        scales *= np.sqrt(2) 
        LML_filter_bank = LM_filters(scales, kernel_size)
        if plot_filters:
            fig, axs = plt.subplots(np.shape(LML_filter_bank)[0],np.shape(LML_filter_bank)[1], figsize = (11,5))
            for i, filter_group in enumerate(LML_filter_bank):
                for j, filter in enumerate(filter_group):
                    filter_normalized  = cv2.normalize(filter,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                    axs[i, j].imshow(filter_normalized, cmap='gray')
                    axs[i, j].axis('off')
            if save_figure:
                plt.savefig('./Code/Results/Filter Banks/LML.png', dpi = 500, bbox_inches="tight")
            plt.show()

        for filter_group in LML_filter_bank:
            for filter in filter_group:
                LM_return_filter.append(filter)
        return LM_return_filter

def gabor_filter(sigma, theta, Lambda, psi, gamma, kernel_size = 49):

    """
    Reference: Wikipedia
    """

    # return cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, Lambda, gamma, psi)
    
    range = int(kernel_size / 2)
    range_spread = np.linspace(-range, range, kernel_size)
    xx, yy= np.meshgrid(range_spread, range_spread)    
    x, y = yy.ravel(), xx.ravel()
    
    x_prime = x * np.cos(theta) + y * np.sin(theta)
    y_prime = x * -np.sin(theta) + y * np.cos(theta)

    exponential = np.exp(-(x_prime ** 2 + (gamma ** 2 * y_prime ** 2)) / (2 * (sigma ** 2)))
    real_cosine = np.cos(2 * np.pi * (x_prime / Lambda) + psi)
    # imaginery_sine = np.sin(2 * np.pi * (x_prime / Lambda) + psi)

    return np.reshape(exponential * real_cosine, (kernel_size, kernel_size))


def get_and_plot_gabor(get_filters = False, plot_filters = False, save_figure = False):
    
    kernel_size = 27
    scales = [3, 4, 6, 8, 10]
    num_theta = 8
    theta_values= np.arange(0, 180, 180/num_theta) * np.pi / 180
    psi = 0
    lambda_values = [3, 4, 5, 8, 11] # for changing lambda
    # Lambda = 5 # for constant lambda
    gamma = 1


    gabor_filter_bank = []
    if plot_filters:
        fig, axs = plt.subplots(len(scales), num_theta, figsize = (11, 5))

    # changing lambda
    for i, (sigma, Lambda) in enumerate(zip(scales, lambda_values)):
        if plot_filters:
            ax_i = axs[i] if len(scales) > 1 else axs
        for j, theta in enumerate(theta_values):
            if plot_filters:
                ax_j = ax_i[j] if num_theta > 1 else ax_i
            filter = gabor_filter(sigma, theta, Lambda, psi, gamma, kernel_size)
            gabor_filter_bank.append(filter)
            if plot_filters:
                filter_normalized = cv2.normalize(filter,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                ax_j.imshow(filter_normalized, cmap='gray')
                ax_j.axis('off')
    
    # constant lambda
    # for i, sigma in enumerate(scales):
    #     if plot_filters:
    #         ax_i = axs[i] if len(scales) > 1 else axs
    #     for j, theta in enumerate(theta_values):
    #         if plot_filters:
    #             ax_j = ax_i[j] if num_theta > 1 else ax_i
    #         filter = gabor_filter(sigma, theta, Lambda, psi, gamma, kernel_size)
    #         gabor_filter_bank.append(filter)
    #         if plot_filters:
    #             filter_normalized = cv2.normalize(filter,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #             ax_j.imshow(filter_normalized, cmap='gray')
    #             ax_j.axis('off')
    if plot_filters:
        if save_figure:
            plt.savefig("./Code/Results/Filter Banks/Gabor.png", dpi=500)
        plt.show()
    if get_filters:
        return gabor_filter_bank

def half_disc_pair(kernel_size, orientation):

    range = int(kernel_size / 2)
    range_spread = np.arange(-range, range+1, 1)
    xx, yy = np.meshgrid(range_spread, range_spread)
    x, y = yy.ravel(), xx.ravel()
    angles = np.arctan2(y, x)
    distances_from_center = np.sqrt(x**2 + y**2)
    
    mask1 = np.zeros(len(range_spread)*len(range_spread))
    mask2 = np.zeros(len(range_spread)*len(range_spread))


    for i, angle in enumerate(angles):
        if distances_from_center[i] <= len(range_spread)/2:
            if (angle >= 0 and angle <= orientation) or (orientation == 0 and angle == np.pi):
                mask1[i] = 255
            if angle < 0 and angle >= -np.pi + orientation:
                mask1[i] = 255

    for i, angle in enumerate(angles):
        if distances_from_center[i] <= len(range_spread)/2:
            if angle >= 0 and angle >= orientation:
                mask2[i] = 255
            if angle < 0 and angle <= -np.pi + orientation:
                mask2[i] = 255


    mask1 = np.reshape(mask1,[len(range_spread),len(range_spread)])
    mask2 = np.reshape(mask2,[len(range_spread),len(range_spread)])
    return mask1, mask2


def get_and_plot_half_masks(get_filters = False, plot_filters = False, save_figure = False):

    scales = [9,15,25]
    orientations = np.array([0, 20, 40, 60, 90, 107, 130, 145]) * (np.pi / 180)
    # orientations = np.linspace(0, 145, 8) * (np.pi / 180)
    left_masks = []
    right_masks = []

    if plot_filters:
        fig, axs = plt.subplots(len(scales) * 2, int(len(orientations) / 2) * 2, figsize = (9, 7))
    for i, size in enumerate(scales):
        if plot_filters:
            ax = axs[2*i]
        for j, orientation in enumerate(orientations[:int(len(orientations)/2)]):
            mask1, mask2 = half_disc_pair(size, orientation) 
            left_masks.append(mask1)
            right_masks.append(mask2)
            if plot_filters:
                mask1_normalized = cv2.normalize(mask1,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                ax[2*j].imshow(mask1_normalized, cmap='gray')
                ax[2*j].axis('off')
                mask2_normalized = cv2.normalize(mask2,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                ax[2*j+1].imshow(mask2_normalized, cmap='gray')
                ax[2*j+1].axis('off')

        if plot_filters:
            ax = axs[2*i + 1]
        for j, orientation in enumerate(orientations[int(len(orientations)/2):]):
            mask1, mask2 = half_disc_pair(size, orientation) 
            left_masks.append(mask1)
            right_masks.append(mask2)
            if plot_filters:
                mask1_normalized = cv2.normalize(mask1,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                ax[2*j].imshow(mask1_normalized, cmap='gray')
                ax[2*j].axis('off')
                mask2_normalized = cv2.normalize(mask2,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                ax[2*j+1].imshow(mask2_normalized, cmap='gray')
                ax[2*j+1].axis('off')
    if plot_filters:
        if save_figure:
            plt.savefig('./Code/Results/Filter Banks/HalfDiscMasks.png', dpi=500)
        plt.show()

        
    if get_filters:
        return [left_masks, right_masks]

def cluster_filter_responses(filter_responses, clusters):

    image_shape = np.shape(filter_responses)
    pixel_feature = np.transpose(filter_responses, axes = [1, 2, 0]).reshape(image_shape[1] * image_shape[2], image_shape[0])

    # init='random',n_init=10,max_iter=300,tol=1e-04
    kmeans = KMeans(n_clusters = clusters, init='random',n_init=10,max_iter=300,tol=1e-04)
    cluster_output = kmeans.fit_predict(pixel_feature)

    return np.reshape(cluster_output, image_shape[1:])

def get_image_gradient(image, bins, half_disc_masks):
    
    left_masks = half_disc_masks[0]
    right_masks = half_disc_masks[1]

    chi_square_distances = []

    for i in range(len(left_masks)):
        chi_square_distance = (image * 0).astype(float)
        for bin in range(bins):
            bin_image = np.float32(image == bin)

            gi = cv2.filter2D(bin_image, ddepth = -1, kernel = np.float32(left_masks[i]))
            hi = cv2.filter2D(bin_image, ddepth = -1, kernel = np.float32(right_masks[i]))

            chi_square_distance += ((gi - hi) ** 2) / (2 * (gi + hi + 1e-10))  

        chi_square_distances.append(chi_square_distance)

    chi_square_distances = np.array(chi_square_distances)
    return np.mean(chi_square_distances, axis = 0)



def main():

    # getting the input images
    images_folder = "./BSDS500/Images/"
    images_list = os.listdir(images_folder)
    images_list.append(images_list.pop(1)) # to put the 10th image at the end of the list

    # creating the folder for storing all the results
    results_folder = "./Code/Results"
    if not os.path.exists(results_folder):
        os.mkdir("./Code/Results")
    
    sobel_baseline_path = "./BSDS500/SobelBaseline/"
    canny_baseline_path = "./BSDS500/CannyBaseline/"


    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """
    DoG = get_and_plot_oriented_DoG(get_filters=True)

    """
    Generate Leung-Malik Filter Bank: (LM)
    Display all the filters in this filter bank and save image as LM.png,
    use command "cv2.imwrite(...)"
    """
    LM_small = get_and_plot_LM(get_filters='small')
    LM_large = get_and_plot_LM(get_filters='large')

    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """
    Gabor = get_and_plot_gabor(get_filters = True)

    filter_bank = [*DoG, *LM_small, *LM_large, *Gabor]

    """
    Generate Half-disc masks
    Display all the Half-disc masks and save image as HDMasks.png,
    use command "cv2.imwrite(...)"
    """
    half_disc_masks = get_and_plot_half_masks(get_filters = True)

    for i, image_name in enumerate(images_list):
        """
        Generate Texton Map
        Filter image using oriented gaussian filter bank
        """
        image_path = images_folder + image_name
        image = cv2.imread(image_path)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image_results_folder = results_folder + f'/{i+1}'
        if not os.path.exists(image_results_folder):
            os.mkdir(image_results_folder)
        
        filter_results = []
        for filter in filter_bank:
            filter_result = cv2.filter2D(grayscale_image, ddepth=-1, kernel=filter)
            filter_results.append(filter_result)
    
        """
        Generate texture ID's using K-means clustering
        Display texton map and save image as TextonMap_ImageName.png,
        use command "cv2.imwrite('...)"
        """

        print(f'Generating texton map for image {i+1}')
        texton_map = cluster_filter_responses(filter_results, clusters=64)
        texton_map_path = image_results_folder + f"/texton_map_{i+1}.png" 
        plt.imshow(texton_map, cmap="nipy_spectral")
        plt.axis('off')
        plt.savefig(texton_map_path, dpi=300, bbox_inches="tight")
        # plt.show()
        
        """
        Generate Texton Gradient (Tg)
        Perform Chi-square calculation on Texton Map
        Display Tg and save image as Tg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        print(f'Generating texton gradient map for image {i+1}')
        texton_gradient = get_image_gradient(texton_map, 64, half_disc_masks)
        texton_gradient_path = image_results_folder + f"/texton_gradient_{i+1}.png" 
        plt.imshow(texton_gradient, cmap="turbo")
        plt.axis('off')
        plt.savefig(texton_gradient_path, dpi=300, bbox_inches="tight")
        # plt.show()

        """
        Generate Brightness Map
        Perform brightness binning 
        """
        print(f'Generating brightness map for image {i+1}')
        brightness_map = cluster_filter_responses(grayscale_image.reshape((1, grayscale_image.shape[0], grayscale_image.shape[1])),16)
        brightness_map_path = image_results_folder + f"/brightness_map_{i+1}.png" 
        plt.imshow(brightness_map)
        plt.axis('off')
        plt.savefig(brightness_map_path, dpi=300, bbox_inches="tight")
        # plt.show()

        """
        Generate Brightness Gradient (Bg)
        Perform Chi-square calculation on Brightness Map
        Display Bg and save image as Bg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        print(f'Generating brightness gradient map for image {i+1}')
        brightness_gradient = get_image_gradient(brightness_map, 16, half_disc_masks)
        brightness_gradient_path = image_results_folder + f"/brightness_gradient_{i+1}.png" 
        plt.imshow(brightness_gradient, cmap="turbo")
        plt.axis('off')
        plt.savefig(brightness_gradient_path, dpi=300, bbox_inches="tight")
        # plt.show()

        """
        Generate Color Map
        Perform color binning or clustering
        """
        print(f'Generating color map for image {i+1}')
        color_map = cluster_filter_responses(np.transpose(image, axes = [2, 0, 1]), clusters = 16)
        color_map_path = image_results_folder + f"/color_map_{i+1}.png" 
        plt.imshow(color_map, cmap="rainbow")
        plt.axis('off')
        plt.savefig(color_map_path, dpi=300, bbox_inches="tight")
        # plt.show()

        """
        Generate Color Gradient (Cg)
        Perform Chi-square calculation on Color Map
        Display Cg and save image as Cg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        print(f'Generating color gradient map for image {i+1}')
        color_gradient = get_image_gradient(color_map, 16, half_disc_masks)
        color_gradient_path = image_results_folder + f"/color_gradient_{i+1}.png" 
        plt.imshow(brightness_gradient, cmap="turbo")
        plt.axis('off')
        plt.savefig(color_gradient_path, dpi=300, bbox_inches="tight")
        # plt.show()

        """
        Read Sobel Baseline
        use command "cv2.imread(...)"
        """
        
        sobel_image_path = sobel_baseline_path + f"/{i+1}.png"
        sobel_baseline = cv2.imread(sobel_image_path)
        sobel_baseline = cv2.cvtColor(sobel_baseline, cv2.COLOR_RGB2GRAY)

        """
        Read Canny Baseline
        use command "cv2.imread(...)"
        """
        canny_image_path = canny_baseline_path + f"/{i+1}.png"
        canny_baseline = cv2.imread(canny_image_path)
        canny_baseline = cv2.cvtColor(canny_baseline, cv2.COLOR_RGB2GRAY)


        """
        Combine responses to get pb-lite output
        Display PbLite and save image as PbLite_ImageName.png
        use command "cv2.imwrite(...)"
        """
        pb_lite_output = np.multiply((0.33*texton_gradient + 0.33*color_gradient + 0.33*brightness_gradient), (0.5*sobel_baseline + 0.5 * canny_baseline))
        pb_lite_path = image_results_folder + f"/pb_lite_output_{i+1}.png" 
        plt.imshow(pb_lite_output, cmap="gray")
        plt.axis('off')
        plt.savefig(pb_lite_path, dpi=300, bbox_inches="tight")
        # plt.show()

        print('\n\n')
    
if __name__ == '__main__':
    main()
 


