import coppafish_tools.if_registration.base as base
import numpy as np
from scipy.ndimage import affine_transform


# The following functions will allow us to create test data
def generate_random_image(spot_dims: list, spot_spread: int, n_spots: int, image_size: list,
                          seed: int) -> np.ndarray:
    """
    Generate a random image with gaussian spots of a given size and number
    :param spot_dims: int size of spots
    :param spot_spread: int spread of spots
    :param n_spots: int number of spots
    :param image_size: list size of image in zyx
    :param seed: int seed for random number generator
    :return: image: np.ndarray image with spots
    """
    np.random.seed(seed)
    image = np.zeros(image_size)
    spot = gaussian_kernel(spot_dims, sigma=spot_spread)
    z_loc = np.random.randint(0, image_size[0] - spot_dims[0], size=n_spots)
    y_loc = np.random.randint(0, image_size[1] - spot_dims[1], size=n_spots)
    x_loc = np.random.randint(0, image_size[2] - spot_dims[2], size=n_spots)
    for i in range(n_spots):

        blc = np.array([z_loc[i], y_loc[i], x_loc[i]])
        trc = blc + spot_dims
        image[blc[0]:trc[0], blc[1]:trc[1], blc[2]:trc[2]] += spot
    image = np.clip(image, a_min=0, a_max=1, out=image)
    image *= 65535
    image = image.astype(np.uint16)

    return image


def gaussian_kernel(size: list, sigma: float) -> np.ndarray:
    """
    Generate a gaussian kernel
    :param size: list size of kernel in zyx
    :param sigma: float sigma of gaussian
    :return: kernel: np.ndarray kernel
    """
    kernel = np.zeros(size)
    for z in range(size[0]):
        for y in range(size[1]):
            for x in range(size[2]):
                kernel[z, y, x] = np.exp(-((z-size[0]/2)**2 + (y-size[1]/2)**2 + (x-size[2]/2)**2)/(2*sigma**2))
    return kernel


def test_register_if():
    im_target = generate_random_image([5, 10, 10], 1, 100, [10, 100, 100], 0)
    affine_transform_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    affine_transform_true[1:3, 1:3] = np.array([[0.9, 0.1], [-0.1, 0.9]])
    affine_transform_true[:3, 3] = np.array([5, 6, 7])
    im_base = affine_transform(im_target, affine_transform_true, order=1)
    affine_transform_test = base.register_if(im_base, im_target)
    assert np.allclose(affine_transform_test, affine_transform_true, atol=0.01)


def test_split_3d_image():
    im = np.random.rand(30, 125, 125)
    subvol_size = [10, 50, 50]
    overlap = 0.1
    subvols, position = base.split_3d_image(im, subvol_size, overlap)
    assert subvols.shape == (12, 10, 50, 50), f"subvols shape: {subvols.shape}"
    assert position.shape == (12, 3), f"position shape: {position.shape}"


def test_procrustes_regression():
    """
    Test the procrustes regression function by givving it a known transformation and checking the output
    """
    shift = np.array([1, 4])
    base_points = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])
    # rotate the points by 90 degrees
    target_points = np.array([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]])
    # shift the points by 1, 4
    target_points += shift
    # run the procrustes regression
    transform = base.procrustes_regression(base_points, target_points)
    # since this function returns a 3 x 4 matrix, we need to extract the rotation and shift
    transform = np.vstack((transform[:2, :2], transform[:2, 3]))
    # check the transform
    assert np.allclose(transform, np.array([[0, -1], [1, 0], shift]), atol=0.01)

