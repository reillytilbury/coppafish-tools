import os
import nd2
import numpy as np
import napari
from tqdm import tqdm
from coppafish import Notebook
from coppafish.register.base import find_shift_array, huber_regression
from coppafish.register.preprocessing import custom_shift, split_3d_image
from coppafish.utils.nd2 import get_nd2_tile_ind, get_metadata
from coppafish.utils.raw import load_image, load_dask
from coppafish.setup.tile_details import get_tilepos
from scipy.ndimage import affine_transform
from skimage.io import imsave
from typing import Tuple


def extract_raw(raw_dir: str, output_dir: str, if_round_name: str, use_channels: list, num_rotations: int = 0,
                overlap: float = 0.1):
    """
    Extract images from ND2 file and save them as .npy files without any filtering
    Args:
        raw_dir: (Str) The directory of the raw data as an ND2 file
        output_dir: (Str) The directory where the images are saved. This should be a folder that is created beforehand.
        if_round_name: (str) Name of the IF round. Images will be saved as if_round_name_t_{tile}c_{channel}.npy
        use_tiles: (list) List of tiles to use
        use_channels: (list) List of channels to use
        num_rotations: (int) Number of rotations to apply to the images (default = 0) rotations are applied in the
            order of axes (1, 2) i.e. y and x axes
        overlap: expected overlap between tiles

    """
    # Check if directories exist
    assert os.path.isfile(raw_dir), f"Raw data file {raw_dir} does not exist"
    assert os.path.isdir(output_dir), f"Output directory {output_dir} does not exist"

    # Load ND2 file
    with nd2.ND2File(raw_dir) as f:
        nd2_file = f.to_dask()
        n_tiles = f.sizes['P']
        n_channels = f.sizes['C']
        tile_sz = f.sizes['X']
        pixel_size_xy = f.metadata.channels[0].volume.axesCalibration[0]

        xy_pos = np.array([f.experiment[0].parameters.points[i].stagePositionUm[:2]
                           for i in range(n_tiles)])
        xy_pos = (xy_pos - np.min(xy_pos, 0)) / pixel_size_xy

    tilepos_yx_nd2, tilepos_yx = get_tilepos(xy_pos=xy_pos, tile_sz=tile_sz, expected_overlap=overlap)

    # Loop through tiles and channels
    for channel in tqdm(use_channels, desc='Extracting channel'):

        channel_dir = os.path.join(output_dir, f'channel_{channel}')
        os.makedirs(channel_dir, exist_ok=True)

        for tile in tqdm(range(n_tiles), desc="Extracting tile"):
            tile_nd2 = get_nd2_tile_ind(tile, tile_pos_yx_nd2=tilepos_yx_nd2, tile_pos_yx_npy=tilepos_yx)

            # Load image (as z,y,x)
            image = np.array(nd2_file[tile_nd2, :, channel])
            image = np.rot90(image, k=num_rotations, axes=(1, 2))[1:]
            image = image.astype(np.uint16)
            # Save image
            _y, _x = tilepos_yx_nd2[tile_nd2]
            imsave(os.path.join(channel_dir, f"{_x}_{_y}.tif"), image)


def extract_seq_dapi(path, output_dir):
    """
    Function to convert DAPI images from sequencing to tif files for zetastitcher.
    If the final shape is non-rectangular, this function will create empty black tiles.

    Args:
        path: path to Notebook
        output_dir: Directory where to save the DAPI files
    """

    # Load notebook
    nb = Notebook(config_file=path)
    config = nb.get_config()

    anchor_dask = load_dask(nb.file_names, nb.basic_info, r=7)

    for i in tqdm(nb.basic_info.use_tiles):

        image = load_image(nb.file_names, nb.basic_info, t=i, c=0, round_dask_array=anchor_dask)
        image = np.swapaxes(image, -1, 0)

        # image = np.rot90(image, k=config['extract']['num_rotations'], axes=(1, 2))

        # Save image
        _y, _x = nb.basic_info.tilepos_yx[i]
        imsave(os.path.join(output_dir, f"{_x}_{_y}.tif"), image)

    # Finding everything possible coordinates in the bounding rectangle
    ymax, xmax = np.max(nb.basic_info.tilepos_yx, axis=0)
    all_pos = list()
    for y in range(ymax+1):
        for x in range(xmax+1):
            all_pos.append((y, x))

    tilepos = list(map(tuple, nb.basic_info.tilepos_yx))

    if len(all_pos) != len(tilepos):

        # Finding missing tiles
        missing_pos = list()
        for p in all_pos:
            if p not in tilepos:
                missing_pos.append(p)

        print(f'Sequencing image was not rectangular, padding the image with {len(missing_pos)} black tiles')

        im_shape = (len(nb.basic_info.use_z), nb.basic_info.tile_sz, nb.basic_info.tile_sz)

        for p in missing_pos:
            imsave(os.path.join(output_dir, f"{p[1]}_{p[0]}.tif"), np.zeros(im_shape))


# nb = Notebook('/home/servers/zaru/ISS/Izzie/Nami-230907_hTau_73g_NN_ADLR+HR_anti/output/notebook.npz')
# config = nb.get_config()
# if_image_dir = ['/home/servers/zaru/ISS/Izzie/Nami-230907_hTau_73g_NN_ADLR+HR_anti/IF_reg/IF_round.nd2_t_0c_0.npy', '/home/servers/zaru/ISS/Izzie/Nami-230907_hTau_73g_NN_ADLR+HR_anti/IF_reg/IF_round.nd2_t_1c_0.npy', '/home/servers/zaru/ISS/Izzie/Nami-230907_hTau_73g_NN_ADLR+HR_anti/IF_reg/IF_round.nd2_t_2c_0.npy', '/home/servers/zaru/ISS/Izzie/Nami-230907_hTau_73g_NN_ADLR+HR_anti/IF_reg/IF_round.nd2_t_3c_0.npy', '/home/servers/zaru/ISS/Izzie/Nami-230907_hTau_73g_NN_ADLR+HR_anti/IF_reg/IF_round.nd2_t_4c_0.npy', '/home/servers/zaru/ISS/Izzie/Nami-230907_hTau_73g_NN_ADLR+HR_anti/IF_reg/IF_round.nd2_t_5c_0.npy', '/home/servers/zaru/ISS/Izzie/Nami-230907_hTau_73g_NN_ADLR+HR_anti/IF_reg/IF_round.nd2_t_6c_0.npy', '/home/servers/zaru/ISS/Izzie/Nami-230907_hTau_73g_NN_ADLR+HR_anti/IF_reg/IF_round.nd2_t_7c_0.npy', '/home/servers/zaru/ISS/Izzie/Nami-230907_hTau_73g_NN_ADLR+HR_anti/IF_reg/IF_round.nd2_t_8c_0.npy', '/home/servers/zaru/ISS/Izzie/Nami-230907_hTau_73g_NN_ADLR+HR_anti/IF_reg/IF_round.nd2_t_9c_0.npy', '/home/servers/zaru/ISS/Izzie/Nami-230907_hTau_73g_NN_ADLR+HR_anti/IF_reg/IF_round.nd2_t_10c_0.npy', '/home/servers/zaru/ISS/Izzie/Nami-230907_hTau_73g_NN_ADLR+HR_anti/IF_reg/IF_round.nd2_t_11c_0.npy']
# anchor_image_dir = [nb.file_names.tile[t][nb.basic_info.anchor_round][nb.basic_info.dapi_channel] for t in nb.basic_info.use_tiles]
# reg_data_dir = '/home/servers/zaru/ISS/Izzie/Nami-230907_hTau_73g_NN_ADLR+HR_anti/output/registration_data.pkl'
# if_image dir will not be available from the notebook so create a list of paths to the IF images and make sure order
# is correct
# reg_data = align_if_round(config, anchor_image_dir, if_image_dir, reg_data_dir)


# Function to load and save images from the ND2 file for IF rounds
def manual_stitch(tile_dir: list, output_dir: str, use_tiles: list, reg_data: dict, tile_origin: np.ndarray,
                  tile_sz: list):
    """
    Load in images as npy files, apply registration, stitch and save
    Args:
        tile_dir: (list of len n_tiles_use) List of paths to images
        output_dir: (str) Path to save stitched image
        use_tiles: (list of len n_tiles_use) List of tiles to use (npy tile index)
        reg_data: (dict) Dictionary of registration data for each tile
        tile_origin: (np.ndarray) Array of tile origins in yxz coordinates in npy order
        tile_sz: tile size in yxz order
    """
    # convert tile origin axis 1 order to zyx instead of yxz
    tile_sz = np.roll(np.array(tile_sz), shift=1)
    tile_origin = np.roll(tile_origin, shift=1, axis=1)
    # load in the intra tile shifts from the registration data
    shift = np.zeros_like(tile_origin) * np.nan
    shift[use_tiles] = np.array([reg_data[tile]['transform'][0, :, 3] for tile in use_tiles])
    # Convert tile origin to be relative to the minimum tile origin
    tile_origin = tile_origin - np.nanmin(tile_origin, axis=0)
    # Size of the stitched image is the maximum tile origin + tile size + 1
    stitched_image_size = (np.nanmax(tile_origin, axis=0).astype(int) +
                           np.array(tile_sz) +
                           1)
    # Create empty array to store stitched image
    stitched_image = np.zeros(stitched_image_size, dtype=np.uint16)

    # Loop through tiles
    for t in tqdm.tqdm(range(len(use_tiles)), desc="Stitching IF images"):
        # Load image
        image = np.load(tile_dir[t])
        # Apply registration
        image = affine_transform(image, reg_data[use_tiles[t]]['transform'][0, :3, :3])
        # blc means bottom left corner, trc means top right corner. Sometimes, these will be negative or greater than
        # the size of the stitched image. In these cases, we clip the values to be within the range of the stitched
        # image and only take the part of the image that is within the stitched image
        blc_nominal = (tile_origin[use_tiles[t]] - shift[use_tiles[t]]).astype(int)
        trc_nominal = blc_nominal + np.array(tile_sz).astype(int)
        blc = np.clip(blc_nominal, a_min=0, a_max=None)
        trc = np.clip(trc_nominal, a_min=None, a_max=stitched_image_size)
        # Now crop image. To do this, we need to compare blc_nominal and trc_nominal to blc and trc
        offset_blc = blc - blc_nominal
        offset_trc = trc_nominal - trc
        image = image[offset_blc[0]:tile_sz[0]-offset_trc[0],
                      offset_blc[1]:tile_sz[1]-offset_trc[1],
                      offset_blc[2]:tile_sz[2]-offset_trc[2]]

        # Stitch image
        stitched_image[blc[0]:trc[0], blc[1]:trc[1], blc[2]:trc[2]] = image

    # Switch back to yxz order
    stitched_image = np.moveaxis(stitched_image, 0, 2)
    # Save stitched image
    np.save(output_dir, stitched_image)


# with open('/home/reilly/ExternalServers/SH/Christina Maat/ISS Data + Analysis/E-2308-005_WT_6mo/11SEP23_2T_rerun/'
#         'output/IF_overlay/reg_data_correct.pkl',
#         'rb') as f:
#     reg_data = pickle.load(f)
# nb = Notebook('/home/reilly/ExternalServers/SH/Christina Maat/ISS Data + Analysis/E-2308-005_WT_6mo/11SEP23_2T_rerun'
#               '/output/notebook.npz')
# tile_dir = ['/home/reilly/local_datasets/christina_if_overlay/tiles/IFr_488-Homer1_594-Iba1P2y12_647-Cd206_t_5c_0.npy',
#             '/home/reilly/local_datasets/christina_if_overlay/tiles/IFr_488-Homer1_594-Iba1P2y12_647-Cd206_t_6c_0.npy']
# output_dir = '/home/reilly/local_datasets/christina_if_overlay/dapi_stitched.npy'
# tile_origin = nb.stitch.tile_origin
# use_tiles = [5, 6]
# tile_sz = [2304, 2304, 64]
# manual_stitch(tile_dir, output_dir, use_tiles, reg_data, tile_origin, tile_sz)
def register_if(anchor_dapi: np.ndarray, if_dapi: np.ndarray, reg_parameters: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Register IF image to anchor image
    :param anchor_dapi: Stitched large anchor image (nz, ny, nx)
    :param if_dapi: Stitched large IF image (nz, ny, nx)
    :param reg_parameters: Dictionary of registration parameters. Keys are:
        * subvolume_size: np.ndarray, size of subvolumes in each dimension (size_z, size_y, size_x)
        * n_subvolumes: np.ndarray, number of subvolumes in each dimension (n_z, n_y, n_x)
        * r_threshold: float, threshold for correlation coefficient

    :return: reg_data: dict, registration data containing:
        * global_shift: np.ndarray, global shift between anchor and IF images
        * global_angle: float, global angle between anchor and IF images
        * local_affine_transforms: dict, local affine transforms for each synthetic tile
    """
    # Steps are as follows:
    # 1. Manual selection of reference points for shift and rotation correction
    # 2. Local correction for z shifts
    if anchor_dapi.shape != if_dapi.shape:
      z_box_anchor, y_box_anchor, x_box_anchor = np.array(anchor_dapi.shape)
      z_box_if, y_box_if, x_box_if = np.array(if_dapi.shape)
      z_box, y_box, x_box = max(z_box_anchor, z_box_if), max(y_box_anchor, y_box_if), max(x_box_anchor, x_box_if)
      anchor_dapi_full, if_dapi_full = np.zeros((z_box, y_box, x_box)), np.zeros((z_box, y_box, x_box))
      anchor_dapi_full[:z_box_anchor, :y_box_anchor, :x_box_anchor] = anchor_dapi
      if_dapi_full[:z_box_if, :y_box_if, :x_box_if] = if_dapi
      anchor_dapi, if_dapi = anchor_dapi_full, if_dapi_full
      del anchor_dapi_full, if_dapi_full

    # 1. Global correction for shift and rotation
    anchor_dapi_2d = np.max(anchor_dapi, axis=0)
    if_dapi_2d = np.max(if_dapi, axis=0)
    v = napari.Viewer()
    v.add_image(anchor_dapi_2d, name='anchor_dapi', colormap='red', blending='additive')
    v.add_image(if_dapi_2d, name='if_dapi', colormap='green', blending='additive')
    v.add_layer(napari.layers.Points(data=np.array([]), name='anchor_dapi_points', size=5, edge_color=np.zeros((3, 4)),
                                     face_color='white'))
    v.add_layer(napari.layers.Points(data=np.array([]), name='if_dapi_points', size=5, edge_color=np.zeros((3, 4)),
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
    # First, split the images into subvolumes
    z_size, y_size, x_size = reg_parameters['subvolume_size']
    z_n, y_n, x_n = reg_parameters['n_subvolumes']
    anchor_subvolumes, position = split_3d_image(anchor_dapi, z_subvolumes=z_n, y_subvolumes=y_n, x_subvolumes=x_n,
                                       z_box=z_size, y_box=y_size, x_box=x_size)
    if_subvolumes, _ = split_3d_image(if_dapi_aligned_initial, z_subvolumes=z_n, y_subvolumes=y_n, x_subvolumes=x_n,
                                      z_box=z_size, y_box=y_size, x_box=x_size)
    # Now loop through subvolumes and calculate the shifts
    shift, corr = find_shift_array(anchor_subvolumes, if_subvolumes, position,
                                   r_threshold=reg_parameters['r_threshold'])

    # Use these shifts to compute a global affine transform
    transform_3d_correction = huber_regression(shift, position, predict_shift=False)
    # Apply the transform to the IF image
    if_dapi_aligned = affine_transform(if_dapi_aligned_initial, transform_3d_correction, order=0)
    transform = (np.vstack((transform_initial, [0, 0, 0, 1])) @ np.vstack((transform_3d_correction, [0, 0, 0, 1])))[:3, :]

    return if_dapi_aligned, transform


def apply_transform(image: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply a 3d affine transform to an image
    :param image: z y x image to be transformed
    :param transform: 3 x 4 transform in z y x
    :return: image_transformed: np.ndaraay transformed image
    """
    image = affine_transform(image, transform, order=5)
    return image


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


# reg_parameters = {'subvolume_size': np.array([16, 512, 512]),
#                     'n_subvolumes': np.array([3, 5, 5]),
#                     'r_threshold': 0.8}
# anchor_dapi = generate_random_image([21, 51, 51], 10, 500, [30, 1500, 1500],
#                                     seed=51)
# # Sort out a transform which our algorithm should be able to find
# angle = 2 * np.pi / 12
# rotation_matrix = np.array([[1, 0, 0],
#                             [0, np.cos(angle), -np.sin(angle)],
#                             [0, np.sin(angle), np.cos(angle)]])
# # we want to rotate around the centre of the image
# offset = (np.eye(3) - rotation_matrix) @ np.array([0, anchor_dapi.shape[1] // 2, anchor_dapi.shape[2] // 2])
# shift = np.array([0, 10, 20])
# if_dapi = affine_transform(anchor_dapi, rotation_matrix, offset=offset, order=0)
# # Now add some 3d shifts
# z_shift = np.arange(-10, 11)
# num_z_shifts = len(z_shift)
# x_chunk = anchor_dapi.shape[2] // num_z_shifts
# for i, z in enumerate(z_shift):
#     begin_x = i * x_chunk
#     end_x = (i + 1) * x_chunk
#     if_dapi[:, :, begin_x:end_x] = custom_shift(if_dapi[:, :, begin_x:end_x],
#                                                         offset=np.array([z, 0, 0]).astype(int))
# # Now run the registration!
# if_dapi_aligned, transform = register_if(anchor_dapi, if_dapi, reg_parameters)
# # Now check the transform
# viewer = napari.Viewer()
# viewer.add_image(anchor_dapi, name='anchor_dapi', colormap='red', blending='additive')
# viewer.add_image(if_dapi_aligned, name='if_dapi', colormap='green', blending='additive')
# napari.run()
# reg_parameters = {'subvolume_size': np.array([16, 512, 512]),
#                     'n_subvolumes': np.array([3, 5, 5]),
#                     'r_threshold': 0.8}
# anchor_dapi = np.load('')
# if_dapi = np.load('')
# if_dapi_aligned, transform = register_if(anchor_dapi, if_dapi, reg_parameters)
# np.save('', transform)
# np.save('', if_dapi_aligned)
