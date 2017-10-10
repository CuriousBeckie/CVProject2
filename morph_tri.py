'''
  File name: morph_tri.py
  Author:
  Date created:
'''
import numpy as np
from scipy.interpolate import *
from scipy.spatial import *
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
import pickle as pickle


def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
    '''
  	File clarification:
    Image morphing via Triangulation
    - Input im1: target image
    - Input im2: source image
    - Input im1_pts: correspondences coordiantes in the target image
    - Input im2_pts: correspondences coordiantes in the source image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
	'''

    # TODO: Your code here
    # Tips: use Delaunay() function to get Delaunay triangulation;
    # Tips: use tri.find_simplex(pts) to find the triangulation index that pts locates in.

    # Finding intermediate images using given warp_frac value
    intermediate_pts = np.zeros((np.size(im1_pts), 2))
    # warp_frac or 1-warp_frac?
    intermediate_pts = (1 - warp_frac) * im1_pts + warp_frac * im2_pts

    # Compute delaunay triangles for intermediate points and print
    Tri = scipy.spatial.Delaunay(intermediate_pts)
    # Tri.find_simplex(np.array([90, 70]))
    # plt.triplot(intermediate_pts[:,0], intermediate_pts[:,1], Tri.simplices.copy())
    # plt.plot(intermediate_pts[:,0], intermediate_pts[:,1], 'o')
    # plt.imshow(im1)
    # plt.show()

    # Compute delaunay triangles for im1_pts points and print
    Tri_im1 = scipy.spatial.Delaunay(im1_pts)
    # Tri_im1.find_simplex(np.array([90, 70]))
    # plt.triplot(im1_pts[:,0], im1_pts[:,1], Tri_im1.simplices.copy())
    # plt.plot(im1_pts[:,0], im1_pts[:,1], 'o')
    # plt.imshow(im1)
    # plt.show()

    # Defining arrays & finding co-ordinates of corners of triangles
    coord_inter = intermediate_pts[Tri.simplices]
    coord_src = im1_pts[Tri_im1.simplices]
    array_coord = np.zeros([len(coord_inter), 3, 3])
    array_src = np.zeros([len(coord_src), 3, 3])

    # Finding inverse of every coordinate matrix before computation of barycentric coordinates
    for i in range(0, len(coord_inter)):
        a = np.transpose(coord_inter[i, :, :])
        b = np.array([1, 1, 1]).reshape(1, 3)
        array_coord[i, :, :] = np.linalg.inv(np.concatenate((a, b), axis=0))

    # for j in range(0, len(coord_src)):
    #    c = np.transpose(coord_src[j, :, :])
    #    d = np.array([1, 1, 1]).reshape(1, 3)
    #    array_src[j, :, :] = np.concatenate((c, d), axis=0)

    # Finding the barycentric co-ordinates for each pixel by vectorization
    # Defining a meshgrid to find barycentric coordinates
    x = np.arange(0, im1.shape[0])
    y = x.reshape(1, im1.shape[0])
    x, y = np.meshgrid(x, y)
    print x,y
    x = x.flatten().reshape((1, np.size(x)))
    y = y.flatten().reshape((1, np.size(x)))

    # Finding the x-y matrix before barycentric coordinate estimation
    xy_matrix = np.concatenate((x, y), axis=0)
    print xy_matrix
    array_xy = np.ones((1, np.size(x)), dtype=np.int)
    xy_matrix = np.concatenate((xy_matrix, array_xy), axis=0)

    # For every point in matrix find barycentric coordinates
    barycentric_coord = np.zeros((xy_matrix.shape[0], xy_matrix.shape[1]))
    print xy_matrix.shape
    for k in range(0, xy_matrix.shape[1]):
        inv = array_coord[Tri.find_simplex(np.array([xy_matrix[0, k], xy_matrix[1, k]]))]
        albega = np.dot(inv, np.transpose(xy_matrix[:, k]))
        # Find the new x,y,z values for every pixel
        newabc = im1_pts[Tri.simplices[Tri.find_simplex(np.array([xy_matrix[0, k], xy_matrix[1, k]]))]]
        newabc = np.transpose(newabc)
        oness = np.ones((1, 3), dtype=np.int)
        newabc = np.concatenate((newabc, oness), axis = 0)
        barycentric_coord[:, k] = np.dot(newabc, albega)

    # Copy pixel value at source image at the original source image to intermediate image using interpolation
    # interp_ch1 = RectBivariateSpline(np.arange(0, im1.shape[0], 1), np.arange(0, im1.shape[1], 1), im1[:, :, 0])
    # interp_ch2 = RectBivariateSpline(np.arange(0, im1.shape[0], 1), np.arange(0, im1.shape[1], 1), im1[:, :, 1])
    # interp_ch3 = RectBivariateSpline(np.arange(0, im1.shape[0], 1), np.arange(0, im1.shape[1], 1), im1[:, :, 2])

    # Copy pixel value at source image at the original source image to intermediate image using floor values
    l = 0
    new_img = np.zeros((300, 300, 3))
    for i in range(0, im1.shape[0]):
        for j in range(0, im1.shape[1]):
            if (barycentric_coord[0, l] < 300) and (barycentric_coord[1, l] < 300):
                new_img[j][i][0] = im1[(np.floor(barycentric_coord[0, l]), np.floor(barycentric_coord[1, l]), 0)]
                new_img[j][i][1] = im1[(np.floor(barycentric_coord[0, l]), np.floor(barycentric_coord[1, l]), 1)]
                new_img[j][i][2] = im1[(np.floor(barycentric_coord[0, l]), np.floor(barycentric_coord[1, l]), 2)]
            l = l + 1

    for i in range(0, im2.shape[0]):
        for j in range(0, im2.shape[1]):
            if (barycentric_coord[0, l] < 300) and (barycentric_coord[1, l] < 300):
                new_img[j][i][0] = im2[(np.floor(barycentric_coord[0, l]), np.floor(barycentric_coord[1, l]), 0)]
                new_img[j][i][1] = im2[(np.floor(barycentric_coord[0, l]), np.floor(barycentric_coord[1, l]), 1)]
                new_img[j][i][2] = im2[(np.floor(barycentric_coord[0, l]), np.floor(barycentric_coord[1, l]), 2)]
            l = l + 1

    plt.triplot(intermediate_pts[:, 0], intermediate_pts[:, 1], Tri.simplices.copy())
    plt.imshow(new_img)
    plt.show()


def main():
    im1 = np.array(Image.open("Monica.jpg").convert('RGB'))
    im2 = np.array(Image.open("Rachel.jpg").convert('RGB'))
    im1_pts = pickle.load(open('save1.p', 'rb'))
    im2_pts = pickle.load(open('save2.p', 'rb'))

    warp_frac = 20.0/60.0
    morph_tri(scipy.misc.imresize(im1,[300,300]), scipy.misc.imresize(im1,[300,300]), im1_pts, im2_pts, np.array([warp_frac, warp_frac]), np.array([warp_frac, warp_frac]))

    # while warp_frac <= 1:
    #   morph_tri(scipy.misc.imresize(im1,[300,300]), scipy.misc.imresize(im1,[300,300]), im1_pts, im2_pts, np.array([warp_frac, warp_frac]), np.array([warp_frac, warp_frac]))
    #   warp_frac = warp_frac + (1/60)

if __name__ == "__main__":
    main()
