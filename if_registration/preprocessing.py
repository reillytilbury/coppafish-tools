import numpy as np
import skimage


def custom_shift(array: np.ndarray, offset: np.ndarray, constant_values=0):
    """
    Compute array shifted by a certain offset.

    Args:
        array: array to be shifted.
        offset: shift value (must be int).
        constant_values: This is the value used for points outside the boundaries after shifting.

    Returns:
        new_array: array shifted by offset.
    """
    array = np.asarray(array)
    offset = np.atleast_1d(offset)
    assert len(offset) == array.ndim
    new_array = np.empty_like(array)

    def slice1(o):
        return slice(o, None) if o >= 0 else slice(0, o)

    new_array[tuple(slice1(o) for o in offset)] = array[tuple(slice1(-o) for o in offset)]

    for axis, o in enumerate(offset):
        new_array[(slice(None),) * axis + (slice(0, o) if o >= 0 else slice(o, None),)] = constant_values

    return new_array


def merge_subvols(position, subvol):
    """
    Suppose we have a known volume V split into subvolumes. The position of the subvolume corner
    in the coords of the initial volume is given by position. However, these subvolumes may have been shifted
    so position may be slightly different. This function finds the minimal volume containing all these
    shifted subvolumes.

    If regions overlap, we take the values from the later subvolume.
    Args:
        position: n_subvols x 3 array of positions of bottom left of subvols (zyx)
        subvol: n_subvols x z_box x y_box x x_box array of subvols

    Returns:
        merged: merged image (size will depend on amount of overlap)
    """
    position = position.astype(int)
    # set min values to 0
    position -= np.min(position, axis=0)
    z_box, y_box, x_box = subvol.shape[1:]
    centre = position + np.array([z_box // 2, y_box // 2, x_box // 2])
    # Get the min and max values of the position, use this to get the size of the merged image and initialise it
    max_pos = np.max(position, axis=0)
    merged = np.zeros((max_pos + subvol.shape[1:]).astype(int))
    neighbour_im = np.zeros_like(merged)
    # Loop through the subvols and add them to the merged image at the correct position.
    for i in range(position.shape[0]):
        subvol_i_mask = np.ix_(
            range(position[i, 0], position[i, 0] + z_box),
            range(position[i, 1], position[i, 1] + y_box),
            range(position[i, 2], position[i, 2] + x_box),
        )
        neighbour_im[subvol_i_mask] += 1
        merged[subvol_i_mask] = subvol[i]

    # identify overlapping regions
    overlapping_pixels = np.argwhere(neighbour_im > 1)
    if len(overlapping_pixels) == 0:
        return merged
    centre_dist = np.linalg.norm(overlapping_pixels[:, None, :] - centre[None, :, :], axis=2)
    # get the index of the closest centre
    closest_centre = np.argmin(centre_dist, axis=1)
    # now loop through subvols and assign overlapping pixels to the closest centre
    for i in range(position.shape[0]):
        subvol_i_pixel_ind = np.where(closest_centre == i)[0]
        subvol_i_pixel_coords_global = np.array([overlapping_pixels[j] for j in subvol_i_pixel_ind])
        subvol_i_pixel_coords_local = subvol_i_pixel_coords_global - position[i]
        z_global, y_global, x_global = subvol_i_pixel_coords_global.T
        z_local, y_local, x_local = subvol_i_pixel_coords_local.T
        merged[z_global, y_global, x_global] = subvol[i, z_local, y_local, x_local]

    return merged


def window_image(image: np.ndarray) -> np.ndarray:
    """
    Window the image by a hann window in y and x and a Tukey window in z.

    Args:
        image: image to be windowed. (z, y, x)

    Returns:
        image: windowed image.
    """
    window_yx = skimage.filters.window("hann", image.shape[1:])
    window_z = signal.windows.tukey(image.shape[0], alpha=0.33)
    if (window_z == 0).all():
        window_z[...] = 1
    window = window_z[:, None, None] * window_yx[None, :, :]
    image = image * window
    return image

