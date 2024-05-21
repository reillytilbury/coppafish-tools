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
from coppafish.register.preprocessing import split_3d_image
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
            * n_subvolumes: np.ndarray, number of subvolumes in each dimension (n_z, n_y, n_x)
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
        z_size, y_size, x_size = 16, 1024, 1024
        nz, ny, nx = np.array(anchor_dapi.shape) // np.array([z_size, y_size, x_size])
        nz, ny, nx = nz + 1, ny + 1, nx + 1
        reg_parameters = {'registration_type': 'subvolume',  # 'shift' or 'subvolume'
                          'subvolume_size': np.array([z_size, y_size, x_size]),
                          'n_subvolumes': np.array([nz, ny, nx]),
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
    assert len(base_points) == len(target_points), "Number of anchor points must equal number of IF points"
    # Calculate the affine transform
    base_mean, target_mean = np.mean(base_points, axis=0), np.mean(target_points, axis=0)
    base_points_centred = base_points - base_mean
    target_points_centred = target_points - target_mean
    U, S, Vt = np.linalg.svd(target_points_centred.T @ base_points_centred)
    R = U @ Vt
    angle = np.arccos(R[0, 0])
    shift = target_mean - base_mean
    # This shift is assuming the affine transform is centred at the centre of mass of the anchor points (base mean), so
    # we need to correct for this and make our shift relative to (0, 0)
    shift += (np.eye(2) - R) @ base_mean
    transform_initial = np.eye(3, 4)
    transform_initial[1:3, 1:3] = R
    transform_initial[1:, 3] = shift
    print(f"Initial angle is {np.round(angle * 180 / np.pi, 2)} degrees and shift is "
          f"{np.round(transform_initial[:2, 2], 2)}")
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
        nz, ny, nx = reg_parameters['n_subvolumes']
        anchor_subvolumes, position = split_3d_image(anchor_dapi, z_subvolumes=nz, y_subvolumes=ny, x_subvolumes=nx,
                                                     z_box=z_size, y_box=y_size, x_box=x_size)
        if_subvolumes, _ = split_3d_image(if_dapi_aligned_initial, z_subvolumes=nz, y_subvolumes=ny, x_subvolumes=nx,
                                          z_box=z_size, y_box=y_size, x_box=x_size)
        # Now loop through subvolumes and calculate the shifts
        shift, corr = find_shift_array(anchor_subvolumes, if_subvolumes, position,
                                       r_threshold=reg_parameters['r_threshold'])

        # Use these shifts to compute a global affine transform
        transform_3d_correction = huber_regression(shift, position, predict_shift=False)

    # Now join the initial and 3d correction transforms
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
    with open(os.path.join(zeta_dir, 'stitch.yaml'), 'r') as f:
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