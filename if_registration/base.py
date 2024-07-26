import os
import nd2
import numpy as np
import napari
import skimage
import tifffile
import yaml
from tqdm import tqdm
from coppafish import Notebook
from coppafish.register.base import find_shift_array, huber_regression
from coppafish.utils.nd2 import get_nd2_tile_ind
from coppafish.utils import tiles_io
from scipy.ndimage import affine_transform


def extract_raw(nb: Notebook, read_dir: str, save_dir: str, use_tiles: list, use_channels: list):
    """
    Extract images from ND2 file and save them as .tif files without any filtering
    Args:
        nb: (Notebook) Notebook of the initial experiment
        read_dir: (Str) The directory of the raw data as an ND2 file
        save_dir: (Str) The directory where the images are saved.
        use_tiles: (list) List of tiles to use
        use_channels: (list) List of channels to use

    """
    if type(use_channels) == int:
        use_channels = [use_channels]
    # Check if directories exist
    assert os.path.isfile(read_dir), f"Raw data file {read_dir} does not exist"
    save_dirs = [save_dir]
    save_dirs += [os.path.join(save_dir, "if", f"channel_{c}") for c in use_channels]
    save_dirs += [os.path.join(save_dir, "seq", f"channel_{nb.basic_info.dapi_channel}")]
    for d in save_dirs:
        if not os.path.isdir(d):
            os.makedirs(d)
    del save_dirs

    # Get NPY and ND2 indices
    tilepos_yx, tilepos_yx_nd2 = nb.basic_info.tilepos_yx, nb.basic_info.tilepos_yx_nd2
    nd2_indices = [get_nd2_tile_ind(t, tilepos_yx_nd2, tilepos_yx) for t in range(nb.basic_info.n_tiles)]
    # get n_rotations
    num_rotations = nb.get_config()['filter']['num_rotations']
    c_dapi = nb.basic_info.dapi_channel

    # Load ND2 file
    with nd2.ND2File(read_dir) as f:
        nd2_file = f.to_dask()

    # 1. collect extracted dapi from seq images
    for t in tqdm(use_tiles, desc="Extracting DAPI from seq images", total=len(use_tiles)):
        y, x = tilepos_yx[t]
        save_path = os.path.join(save_dir, "seq", f"channel_{c_dapi}", f"{x}_{y}.tif")
        if os.path.isfile(save_path):
            continue
        # load raw image
        raw_path = nb.file_names.tile_unfiltered[t][nb.basic_info.anchor_round][c_dapi]
        image_raw = tiles_io._load_image(raw_path, nb.extract.file_type)
        # apply rotation for versions less than 0.11.0
        version = [int(i) for i in nb.extract.software_version.split('.')]
        if not (version[0] >= 1 or version[1] >= 11):
            image_raw = np.rot90(image_raw, k=num_rotations, axes=(1, 2))
        # Save image in the format x_y.tif
        tifffile.imwrite(save_path, image_raw)

    # 2. extract all relevant channels from the IF images
    for t in tqdm(use_tiles, desc="Extracting IF images", total=len(use_tiles)):
        for c in use_channels:
            y, x = tilepos_yx[t]
            save_path = os.path.join(save_dir, "if", f"channel_{c}", f"{x}_{y}.tif")
            if os.path.isfile(save_path):
                continue
            # load image
            image = np.array(nd2_file[nd2_indices[t], :, c])
            image = np.rot90(image, k=num_rotations, axes=(1, 2))[1:]
            image = image.astype(np.uint16)
            # Save image in the format x_y.tif
            tifffile.imwrite(save_path, image)


def register_if(anchor_dapi: np.ndarray,
                if_dapi: np.ndarray,
                transform_save_dir: str,
                reg_parameters: dict = None,
                downsample_factor_yx: int = 4) -> np.ndarray:
    """
    Register IF image to anchor image
    :param anchor_dapi: Stitched large anchor image (nz, ny, nx)
    :param if_dapi: Stitched large IF image (nz, ny, nx)
    :param transform_save_dir: str, directory to save the transform as a .npy file
    :param reg_parameters: Dictionary of registration parameters. Keys are:
        * registration_type: str, type of registration to perform (must be 'shift' or 'subvolume')
        if registration_type is 'shift':
            No additional parameters are required
        if registration_type is 'subvolume':
            * subvolume_size: np.ndarray, size of subvolumes in each dimension (size_z, size_y, size_x)
            * overlap: float, fraction of overlap between subvolumes: 0 <= overlap < 1
            * r_threshold: float, threshold for correlation coefficient
    :param downsample_factor_yx: int, downsample factor for y and x dimensions


    :return:
        transform: np.ndarray, affine transform matrix
    """
    # Steps are as follows:
    # 1. Manual selection of reference points for shift and rotation correction
    # 2. Local correction for z shifts (done as a global shift correction or by subvolume registration)

    if anchor_dapi.shape != if_dapi.shape:
        z_box_anchor, y_box_anchor, x_box_anchor = np.array(anchor_dapi.shape)
        z_box_if, y_box_if, x_box_if = np.array(if_dapi.shape)
        z_box, y_box, x_box = max(z_box_anchor, z_box_if), max(y_box_anchor, y_box_if), max(x_box_anchor, x_box_if)
        anchor_dapi_full, if_dapi_full = np.zeros((z_box, y_box, x_box)), np.zeros((z_box, y_box, x_box))
        anchor_dapi_full[:z_box_anchor, :y_box_anchor, :x_box_anchor] = anchor_dapi
        if_dapi_full[:z_box_if, :y_box_if, :x_box_if] = if_dapi
        anchor_dapi, if_dapi = anchor_dapi_full, if_dapi_full
        del anchor_dapi_full, if_dapi_full

    if reg_parameters is None:
        z_size, y_size, x_size = 16, 512, 512
        reg_parameters = {'registration_type': 'subvolume',  # 'shift' or 'subvolume'
                          'subvolume_size': [z_size, y_size, x_size],
                          'overlap': 0.1,
                          'r_threshold': 0.8}

    # 1. Global correction for shift and rotation using procrustes analysis
    anchor_dapi_2d = np.max(anchor_dapi, axis=0)
    if_dapi_2d = np.max(if_dapi, axis=0)
    v = napari.Viewer()
    v.add_image(anchor_dapi_2d, name='anchor_dapi', colormap='red', blending='additive')
    v.add_image(if_dapi_2d, name='if_dapi', colormap='green', blending='additive')
    v.add_layer(napari.layers.Points(data=np.array([]), name='anchor_dapi_points', size=1, edge_color=np.zeros((3, 4)),
                                     face_color='white'))
    v.add_layer(napari.layers.Points(data=np.array([]), name='if_dapi_points', size=1, edge_color=np.zeros((3, 4)),
                                     face_color='white'))
    v.show(block=True)

    # Get user input for shift and rotation
    base_points = v.layers[2].data
    target_points = v.layers[3].data
    # Calculate the original orthogonal transform
    transform_initial = procrustes_regression(base_points, target_points)
    # Now apply the transform to the IF image
    if_dapi_aligned_initial = affine_transform(if_dapi, transform_initial, order=0)

    v = napari.Viewer()
    v.add_image(anchor_dapi, name='anchor_dapi', colormap='red', blending='additive')
    v.add_image(if_dapi_aligned_initial, name='if_dapi', colormap='green', blending='additive')
    v.show(block=True)

    # 2. Local correction for shifts
    if reg_parameters['registration_type'] == 'shift':
        # shift needs to be shift taking anchor to if, as the first transform was obtained this way
        shift = skimage.registration.phase_cross_correlation(reference_image=if_dapi_aligned_initial,
                                                             moving_image=anchor_dapi)[0]
        transform_3d_correction = np.eye(3, 4)
        transform_3d_correction[:, 3] = shift

    elif reg_parameters['registration_type'] == 'subvolume':
        # First, split the images into subvolumes
        z_size, y_size, x_size = reg_parameters['subvolume_size']
        anchor_subvolumes, position = split_3d_image(image=anchor_dapi, subvolume_size=[z_size, y_size, x_size],
                                                     overlap=reg_parameters['overlap'])
        if_subvolumes, _ = split_3d_image(image=if_dapi_aligned_initial, subvolume_size=[z_size, y_size, x_size],
                                          overlap=reg_parameters['overlap'])
        # Now loop through subvolumes and calculate the shifts
        shift, corr = find_shift_array(anchor_subvolumes, if_subvolumes, position,
                                       r_threshold=reg_parameters['r_threshold'])
        # flatten the position array
        position = position.reshape(-1, 3)

        # Use these shifts to compute a global affine transform
        transform_3d_correction = huber_regression(shift, position, predict_shift=False)
    else:
        raise ValueError("Invalid registration type. Must be 'shift' or 'subvolume'")

    # plot the transformed image
    if_dapi_aligned = affine_transform(if_dapi_aligned_initial, transform_3d_correction, order=0)
    v = napari.Viewer()
    v.add_image(anchor_dapi, name='anchor_dapi', colormap='red', blending='additive')
    v.add_image(if_dapi_aligned, name='if_dapi', colormap='green', blending='additive')
    v.show(block=True)

    # Now compose the initial and 3d correction transforms
    transform = (np.vstack((transform_initial, [0, 0, 0, 1])) @
                 np.vstack((transform_3d_correction, [0, 0, 0, 1])))[:3, :]
    # up-sample shift in yx
    transform[1:, -1] *= downsample_factor_yx
    np.save(os.path.join(transform_save_dir, 'transform.npy'), transform)
    return transform


def apply_transform(im_dir: str, transform: np.ndarray, save_dir: str):
    """
    Apply the transform to an image and save the result
    :param im_dir: str, directory of the image to apply the transform to (should be a tiff file)
    :param transform: np.ndarray, affine transform matrix
    :param save_dir: str, directory to save the transformed image
    """
    # Load the image
    image = tifffile.imread(im_dir)
    # Apply the transform
    transformed_image = affine_transform(image, transform, order=0)
    # Save the transformed image
    im_name = os.path.basename(im_dir).split('.')[0] + '_registered.tif'
    save_dir = os.path.join(save_dir, im_name)
    tifffile.imwrite(save_dir, transformed_image)


def convert_notebook_coords_to_zeta(nb: Notebook, zeta_dir: str):
    """
    Convert notebook coordinates to zetastitcher coordinates using the stitch.yaml file in the zeta_dir. This function
    does not return anything, but saves the notebook with the new coordinates under the name notebook_zeta.npz.

        :param nb: Notebook with global coords to be converted
        :param zeta_dir: The directory containing the stitch.yaml file
    """
    # Load the stitch.yaml file
    with open(zeta_dir, 'r') as f:
        stitch = yaml.safe_load(f)
    stitch = stitch['filematrix']

    # get number of tiles and initialise tile_origins_yxz
    tilepos_yx, n_tiles = nb.basic_info.tilepos_yx.copy(), len(stitch)
    assert len(stitch) == len(tilepos_yx), 'Number of tiles in stitch.yaml does not match number of tiles in tilepos_yx'
    tile_origins_yxz = np.zeros((n_tiles, 3)) # yxz
    npy_tile_index = np.zeros(n_tiles, dtype=int) # index of each zetastitcher tile in npy tiles format

    # iterate through each tile in stitch.yaml and find the corresponding tile in tilepos_yx
    for i, tile in enumerate(stitch):
        # get x, y position of tile
        x, y = int(tile['X']), int(tile['Y'])
        # need to find the index of [y, x] in tilepos_yx
        npy_tile_index[i] = np.where((tilepos_yx == [y, x]).all(axis=1))[0][0]

    # iterate through all tiles in stitch.yaml and populate tile_origins_yxz
    for i, tile in enumerate(stitch):
        # get origin of tile
        origin_i_yxz = np.array([tile['Ys'], tile['Xs'], tile['Zs']])
        # assign origin to correct index in tile_origins_yxz
        tile_origins_yxz[npy_tile_index[i]] = origin_i_yxz

    # now replace the tile_origins parameter in the stitch page of the notebook with the new parameter and save the
    # notebook under the new name notebook_zeta
    nb.stitch.finalized = False
    del nb.stitch.tile_origin
    nb.stitch.tile_origin = tile_origins_yxz
    new_nb_name = os.path.join(nb.file_names.output_dir, 'notebook_zeta.npz')
    nb.save(new_nb_name)


def split_3d_image(image: np.ndarray, subvolume_size: list, overlap: float = 0.1):
    """
    Split a 3D image into subvolumes of size given by subvolume_size
    Args:
        image: (np.ndarray) 3D image to split (nz, ny, nx)
        subvolume_size: Size of the subvolumes in each dimension (size_z, size_y, size_x)
        overlap: (float) Fraction of overlap between subvolumes: 0 <= overlap < 1 (default 0.1)

    Returns:
        subvolumes: (np.ndarray) List of subvolumes (n_subvolumes, size_z, size_y, size_x)
        position: (np.ndarray) List of positions of the subvolumes in the original image (n_subvolumes, 3)
    """
    assert 0 <= overlap < 1, "Overlap must be between 0 and 1"
    assert len(subvolume_size) == 3, "Subvolume size must be a list of 3 integers"
    assert all([type(i) == int for i in subvolume_size]), "Subvolume size must be a list of 3 integers"
    assert [subvolume_size[i] <= image.shape[i] for i in range(3)], "Subvolume size must be smaller than image size"

    # determine number of subvolumes in each dimension. This crops the image to the nearest multiple of subvolume_size,
    # after accounting for overlap
    im_size = np.array(image.shape)
    n_subvolumes = (im_size // ((1-overlap) * np.array(subvolume_size))).astype(int)
    # regression will only work if there are more than 1 subvolume in each dimension
    assert all(n_subvolumes > 1), "Subvolume size too large for image size. Reduce subvolume size or increase overlap"

    # create an array of dimensions (n_subvolumes_z, n_subvolumes_y, n_subvolumes_x, size_z, size_y, size_x)
    subvol_dims = np.array([n_subvolumes, subvolume_size]).flatten()
    subvolumes = np.zeros(subvol_dims)
    positions = np.zeros(np.concatenate([n_subvolumes, [3]]))
    # populate the subvolumes array
    for z, y, x in tqdm(np.ndindex(tuple(n_subvolumes)), desc="Splitting image into subvolumes"):
        z_start, y_start, x_start = (np.array([z, y, x]) * (1-overlap) * np.array(subvolume_size)).astype(int)
        z_end, y_end, x_end = z_start + subvolume_size[0], y_start + subvolume_size[1], x_start + subvolume_size[2]
        z_centre, y_centre, x_centre = (z_start + z_end) // 2, (y_start + y_end) // 2, (x_start + x_end) // 2
        subvolumes[z, y, x] = image[z_start:z_end, y_start:y_end, x_start:x_end]
        positions[z, y, x] = np.array([z_centre, y_centre, x_centre])

    return subvolumes, positions


def procrustes_regression(base_points: np.ndarray, target_points: np.ndarray):
    """
    Perform procrustes analysis to find the affine transform between two sets of points. This will return the best
    orthogonal transformation between the two sets of points. This is useful for finding the rotation and translation
    between two sets of points.
    Args:
        base_points: n_points x 2 np.ndarray of points to transform
        target_points: n_points x 2 np.ndarray of points to transform to

    Returns:
        transform_procrustes: 3x4 np.ndarray, affine transform matrix
    """
    assert base_points.shape == target_points.shape, "Base and target points must have the same shape"
    assert base_points.shape[1] == 2, "Base and target points must be 2D"

    # centre the points
    base_mean, target_mean = np.mean(base_points, axis=0), np.mean(target_points, axis=0)
    base_points_centred = base_points - base_mean
    target_points_centred = target_points - target_mean

    # procrustes analysis uses svd to find optimal rotation
    U, S, Vt = np.linalg.svd(target_points_centred.T @ base_points_centred)
    R = U @ Vt
    angle = np.arccos(R[0, 0])
    shift = target_mean - base_mean
    print(f"Initial angle is {np.round(angle * 180 / np.pi, 2)} degrees and shift is {np.round(shift, 2)}")

    # This shift found takes origin at the centre of mass of the anchor points (base mean),
    # correct for this and make our shift relative to (0, 0)
    shift += (np.eye(2) - R) @ base_mean
    # convert our matrix to a 3 x 4 affine transform matrix (as this will be the starting point for 3d registration)
    transform_procrustes = np.eye(3, 4)
    transform_procrustes[1:3, 1:3] = R
    transform_procrustes[1:, 3] = shift
    return transform_procrustes