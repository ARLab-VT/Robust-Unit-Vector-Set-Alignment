import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from numpy.linalg import svd
import pickle
import os

def read_pickle_file(filename, folder_path="./vector_alignment_dataset/"):
    """
    Load data from a pickle file.

    Args:
        filename (str): The name of the pickle file to read.
        folder_path (str): The path to the folder containing the file. Defaults to "./vector_alignment_dataset/".

    Returns:
        The data loaded from the pickle file.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        IOError: If there is an error reading the file.
    """
    # Construct the full file path
    file_path = os.path.join(folder_path, filename)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    # Load and return the data from the pickle file
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    except IOError as e:
        raise IOError(f"An error occurred while reading the file '{file_path}': {e}")
    
    return data

def save_to_pickle(results, folder_path, filename):
    # Ensure the directory exists
    os.makedirs(folder_path, exist_ok=True)

    # Combine the path and filename to create the full file path
    filepath = os.path.join(folder_path, filename)

    # Save the results to the specified pickle file
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

def Rx(theta_deg):
    """
    Creates a rotation matrix for a rotation around the x-axis by a given angle in degrees.

    Args:
        theta_deg (float): The rotation angle in degrees.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix representing the rotation.
    """
    x_rad = np.deg2rad(theta_deg)
    return np.array([[1, 0, 0],
                     [0, np.cos(x_rad), -np.sin(x_rad)],
                     [0, np.sin(x_rad), np.cos(x_rad)]])

def Ry(theta_deg):
    """
    Creates a rotation matrix for a rotation around the y-axis by a given angle in degrees.

    Args:
        theta_deg (float): The rotation angle in degrees.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix representing the rotation.
    """
    y_rad = np.deg2rad(theta_deg)
    return np.array([[np.cos(y_rad), 0, np.sin(y_rad)],
                     [0, 1, 0],
                     [-np.sin(y_rad), 0, np.cos(y_rad)]])

def Rz(theta_deg):
    """
    Creates a rotation matrix for a rotation around the z-axis by a given angle in degrees.

    Args:
        theta_deg (float): The rotation angle in degrees.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix representing the rotation.
    """
    z_rad = np.deg2rad(theta_deg)
    return np.array([[np.cos(z_rad), -np.sin(z_rad), 0],
                     [np.sin(z_rad), np.cos(z_rad), 0],
                     [0, 0, 1]])
    
def R1_R2_degError(R_exp, R_est):
    """
    Calculate angular error in degrees between two rotation matrices R_exp and R_est.
    """
    return np.degrees(abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0))));

def cart_centroid(cart_point_array):
    """
    Computes the centroid (geometric center) of a set of Cartesian points.

    Args:
        cart_point_array (numpy.ndarray): An array of shape (n, m) where `n` is the number of points and `m` is the number of dimensions.

    Returns:
        numpy.ndarray: A 1D array of length `m` representing the centroid of the input points.
    """
    return np.mean(cart_point_array, axis=0)

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Compute the rotation matrix that rotates vector `vec1` to align with vector `vec2`.

    Args:
        vec1 (array-like): A 3-element array representing the initial vector.
        vec2 (array-like): A 3-element array representing the target vector.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix `R` such that `R @ vec1_normalized = vec2_normalized`.

    Raises:
        ValueError: If either `vec1` or `vec2` is a zero vector.
    """
    # Convert input vectors to numpy arrays and ensure they are 3-dimensional
    a = np.array(vec1, dtype=float).reshape(3)
    b = np.array(vec2, dtype=float).reshape(3)
    
    # Normalize the input vectors
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Input vectors must be non-zero.")
    
    a_normalized = a / norm_a
    b_normalized = b / norm_b

    # Compute the cross product and dot product of the normalized vectors
    v = np.cross(a_normalized, b_normalized)
    c = np.dot(a_normalized, b_normalized)
    
    # Compute the sine of the angle between vectors
    s = np.linalg.norm(v)
    
    # Handle the special cases where vectors are parallel or anti-parallel
    if s == 0:
        if c > 0:
            # Vectors are already aligned
            return np.eye(3)
        else:
            # Vectors are opposite; find a rotation of 180 degrees around an arbitrary orthogonal axis
            # Find an orthogonal vector to `a_normalized`
            orthogonal = np.array([1, 0, 0]) if not np.allclose(a_normalized, [1, 0, 0]) else np.array([0, 1, 0])
            v = np.cross(a_normalized, orthogonal)
            v /= np.linalg.norm(v)
            # Use the outer product formula for 180-degree rotation
            rotation_matrix = -np.eye(3) + 2 * np.outer(v, v)
            return rotation_matrix

    # Compute the skew-symmetric cross-product matrix of v
    kmat = np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]])
    
    # Compute the rotation matrix using Rodrigues' rotation formula
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    
    return rotation_matrix

def rotate_a_point_cloud(point_cloud, rotation_matrix):
    """
    Rotate a point cloud using a given rotation matrix.

    Args:
        point_cloud (numpy.ndarray): An array of shape (n, 3), where `n` is the number of points.
                                     Each row represents a point with coordinates (x, y, z).
        rotation_matrix (numpy.ndarray): A 3x3 matrix representing the rotation to be applied.

    Returns:
        numpy.ndarray: The rotated point cloud, with the same shape as the input (n, 3).

    Raises:
        AssertionError: If `point_cloud` does not have shape (n, 3) or if `rotation_matrix` is not 3x3.
    """
    # Ensure the point cloud has the correct shape (n, 3)
    assert point_cloud.shape[1] == 3, "Point cloud must have shape (n, 3)"
    
    # Ensure the rotation matrix is 3x3
    assert rotation_matrix.shape == (3, 3), "Rotation matrix must be 3x3"
    
    # Rotate the point cloud by multiplying with the transpose of the rotation matrix
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix.T)
    
    return rotated_point_cloud



# def cart2sph(xyz_array):
#     """
#     Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).

#     Args:
#         xyz_array (numpy.ndarray): A 1D array of length 3 representing Cartesian coordinates [x, y, z].

#     Returns:
#         numpy.ndarray: A 1D array of length 3 representing spherical coordinates [r, theta, phi],
#                        where r is the radial distance, theta is the polar angle (in radians),
#                        and phi is the azimuthal angle (in radians).
#     """
#     x, y, z = xyz_array
    
#     # Calculate the radial distance (r)
#     r = np.linalg.norm(xyz_array)
    
#     # Calculate the polar angle (theta)
#     theta = np.arccos(z / r)  # angle from the z-axis
    
#     # Calculate the azimuthal angle (phi)
#     phi = np.arctan2(y, x)  # angle in the x-y plane
    
#     return np.array([r, theta, phi])


def cart2sph(xyz_array):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi),
    with phi in the range [0, 2*pi].

    Args:
        xyz_array (numpy.ndarray): A 1D array of length 3 representing Cartesian coordinates [x, y, z].

    Returns:
        numpy.ndarray: A 1D array of length 3 representing spherical coordinates [r, theta, phi],
                       where r is the radial distance, theta is the polar angle (in radians),
                       and phi is the azimuthal angle (in radians).
    """
    x, y, z = xyz_array
    
    # Calculate the radial distance (r)
    r = np.linalg.norm(xyz_array)
    
    # Calculate the polar angle (theta)
    theta = np.arccos(z / r)  # angle from the z-axis
    
    # Calculate the azimuthal angle (phi) and adjust to [0, 2*pi]
    phi = np.arctan2(y, x)  # angle in the x-y plane
    if phi < 0:
        phi += 2 * np.pi  # convert to range [0, 2*pi]
    
    return np.array([r, theta, phi])



def cart2geo(xyz_array):
    """
    Convert Cartesian coordinates (x, y, z) to geographic coordinates (latitude, longitude).

    Args:
        xyz_array (numpy.ndarray): A 1D array of length 3 representing Cartesian coordinates [x, y, z].

    Returns:
        tuple: A tuple (latitude, longitude) in degrees,
               where latitude is in the range [-90, 90] and longitude is in the range [-180, 180].
    """
    # Convert Cartesian to spherical coordinates
    r, theta, phi = cart2sph(xyz_array)
    
    # Convert polar angle (theta) to latitude
    #lat = 90 - np.degrees(theta)
    lat = np.degrees(theta)
    
    # Convert azimuthal angle (phi) to longitude
    lon = np.degrees(phi)
    
    return lat, lon

def cart2geo_array(array_of_cart):
    """
    Convert an array of Cartesian points to geographic coordinates (latitude, longitude) in degrees.

    Args:
        array_of_cart (numpy.ndarray): A 2D array of shape (n, 3) where each row represents
                                       Cartesian coordinates [x, y, z].

    Returns:
        numpy.ndarray: A 2D array of shape (n, 2) where each row contains [latitude, longitude] in degrees.
    """
    # Convert each Cartesian coordinate to geographic coordinates using a list comprehension
    geo_coords = np.array([cart2geo(cart) for cart in array_of_cart])
    
    return geo_coords

def flatten_2d(mat_2d):
    """
    Sums up all rows in each column of a 2D NumPy array to create a 1D array.

    Parameters:
    - mat_2d (numpy.ndarray): A 2D NumPy array of shape (n, m), where `n` is the number of rows and `m` is the number of columns.

    Returns:
    - numpy.ndarray: A 1D NumPy array of shape (m,), where each element is the sum of 
                     the respective column in the input 2D array.
    """
    # Sum the elements along the rows (axis 0) to get the sum of each column
    return np.sum(mat_2d, axis=0)


def hist2D_from_geo_array(geo_array, lat_bin=180, long_bin=360):
    """
    Create a 2D histogram from geographic coordinates.

    Args:
        geo_array (numpy.ndarray): An (n, 2) array where each row represents [latitude, longitude] in degrees.
        lat_bin (int): Number of bins for latitude. Default is 180.
        long_bin (int): Number of bins for longitude. Default is 360.

    Returns:
        numpy.ndarray: A 2D histogram array where the rows correspond to latitude bins and columns to longitude bins.
    """
    # Compute the 2D histogram
    hist, xedges, yedges = np.histogram2d(
        geo_array[:, 1],  # Longitude
        geo_array[:, 0],  # Latitude
        bins=(long_bin, lat_bin),
        #range=[[-180, 180], [-90, 90]],
        range=[[0, 360], [0, 180]],
        density=True
    )
    
    # Flip and transpose the histogram to match geographic orientation
    hist = np.flipud(hist.T)
    
    return hist


def find_1D_shift_wofft(gt_hist, moving_hist):
    """
    Find the optimal circular shift that maximizes the cross-correlation between gt_hist and moving_hist.

    Args:
        gt_hist (numpy.ndarray): The reference histogram (ground truth).
        moving_hist (numpy.ndarray): The histogram to be shifted.

    Returns:
        float: The best shift as a fraction of 360 degrees.
    """
    assert len(gt_hist) == len(moving_hist), "Histograms must have the same length."
    
    k = len(gt_hist)
    n = k / 360  # Conversion factor for shifts to degrees
    
    max_correlation = -np.inf
    best_shift_flat = 0
    
    for shift in range(k):
        # Circular shift the moving_hist
        shifted_hist = np.roll(moving_hist, shift)
        
        # Compute the cross-correlation
        correlation = np.dot(gt_hist, shifted_hist)
        
        # Update the maximum correlation and best shift if needed
        if correlation > max_correlation:
            max_correlation = correlation
            best_shift_flat = shift
    
    # Return the best shift normalized to a fraction of 360 degrees
    return best_shift_flat / n




def generate_random_rotations(n):
    """
    Generate `n` random 3x3 rotation matrices.

    Args:
        n (int): The number of random rotations to generate.

    Returns:
        numpy.ndarray: An array of shape (n, 3, 3) where each element is a 3x3 rotation matrix.
    """
    rotations = [R.random(random_state=np.random.default_rng()).as_matrix() for _ in range(n)]
    return np.array(rotations)

################ Fast Spatial Alignment ###########


def pcl_array_to_rotation_angle_array(points):
    """
    Convert a point cloud array to an array of rotation angles.

    Args:
        points (array-like): An (n, 3) array of 3D points, where n is the number of points.
                             Each point is represented as [x, y, z].

    Returns:
        numpy.ndarray: An (n, 3) array of rotation angles in degrees.
                       Each row represents [theta_x, theta_y, theta_z],
                       where theta_x, theta_y, and theta_z are rotation angles
                       around the X, Y, and Z axes respectively.

    Description:
        This function calculates rotation angles for each point in a 3D point cloud.
        It projects the points onto the XY, XZ, and YZ planes and computes the
        corresponding angles using the arctan2 function. The angles are then
        adjusted to ensure they are in the range [0, 360] degrees.
    """
    # Ensure points is a NumPy array
    points = np.asarray(points)
    
    # Calculate the angles for each projection
    theta_z = adjust_angle(points[:, 1], points[:, 0])  # Rotation around Z-axis from XY projection
    theta_y = adjust_angle(points[:, 0], points[:, 2])  # Rotation around Y-axis from XZ projection
    theta_x = adjust_angle(points[:, 2], points[:, 1])  # Rotation around X-axis from YZ projection
    
    # Stack the angles together into an (n, 3) array
    angles = np.stack((theta_x, theta_y, theta_z), axis=-1)
    return angles


def adjust_angle(y, x):
    """
    Calculate and adjust the angle between two coordinates.

    Args:
        y (numpy.ndarray): Array of y-coordinates.
        x (numpy.ndarray): Array of x-coordinates.

    Returns:
        numpy.ndarray: Array of adjusted angles in degrees.

    Description:
        This function calculates the angle between the positive x-axis and the
        vector from the origin to (x, y) using arctan2. It then adjusts negative
        angles by adding 2π to ensure all angles are in the range [0, 2π].
        Finally, it converts the angles from radians to degrees.
    """
    angle = np.arctan2(y, x)
    angle[angle < 0] += 2 * np.pi
    return np.degrees(angle)

def rotation_angles2mat(x_rot, y_rot, z_rot):
    """
    Convert rotation angles to a 3D rotation matrix.

    Args:
        x_rot (float): Rotation angle around the x-axis in degrees.
        y_rot (float): Rotation angle around the y-axis in degrees.
        z_rot (float): Rotation angle around the z-axis in degrees.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix representing the combined rotation.

    Description:
        This function takes rotation angles for each axis (in degrees) and 
        converts them to a single 3D rotation matrix. The rotations are applied 
        in the order Z-Y-X (first around Z, then Y, then X).

    Note:
        The rotation order can be changed by modifying the matrix multiplication
        order in the final step.
    """

    # Combine the rotations, which is typically done in the order ZYX
    R = (Rz(z_rot)) @ (Ry(y_rot)) @ (Rx(x_rot))

    return R

def hist1D(array1D, number_of_bins, range_min, range_max):
    """
    Create a 1D histogram from the given array.

    Args:
        array1D (array-like): The input array to be histogrammed.
        number_of_bins (int): The number of bins to use in the histogram.
        range_min (float): The minimum value of the range to be histogrammed.
        range_max (float): The maximum value of the range to be histogrammed.

    Returns:
        numpy.ndarray: An array of integers representing the histogram counts.

    Description:
        This function creates a 1D histogram from the input array. It uses numpy's
        histogram function to bin the data into the specified number of bins
        within the given range. The function returns only the histogram counts,
        not the bin edges.

    Note:
        Values outside the specified range are not included in the histogram.
    """
    # Create the histogram
    histogram, bin_edges = np.histogram(array1D, bins=number_of_bins, range=(range_min, range_max))
    return histogram

def estimate_rotation_one_to_one(destination, source):
    """
    Estimate the rotation matrix that aligns the source point cloud to the destination point cloud.
    
    This function assumes that the two point clouds are related by a rotation and translation only.
    The rotation matrix is estimated using the Singular Value Decomposition (SVD) method.

    Args:
    - destination (np.ndarray): An (n, 3) array representing the destination point cloud.
    - source (np.ndarray): An (n, 3) array representing the source point cloud.

    Returns:
    - np.ndarray: A (3, 3) rotation matrix that best aligns the source point cloud to the destination point cloud.
    """
    
    assert destination.shape == source.shape, "Point clouds must have the same shape (n, 3)"
    
    # Calculate the centroids of the point clouds
    centroid_dest = np.mean(destination, axis=0)
    centroid_source = np.mean(source, axis=0)
    
    # Center the point clouds by subtracting the centroids
    centered_dest = destination - centroid_dest
    centered_source = source - centroid_source
    
    # Calculate the covariance matrix
    cov_matrix = np.dot(centered_source.T, centered_dest)
    
    # Perform SVD on the covariance matrix
    U, _, Vt = svd(cov_matrix)
    
    # Compute the rotation matrix
    rotation_matrix = np.dot(U, Vt)
    
    # Ensure the matrix is a proper rotation matrix (det = 1)
    if np.linalg.det(rotation_matrix) < 0:
        Vt[-1, :] *= -1  # Correct for reflection by flipping the sign of the last row of Vt
        rotation_matrix = np.dot(U, Vt)
    
    return rotation_matrix.T

########### Algorithms ###########
def SPMC(dst, src, lat_bin=180, long_bin=360, use_fft = False):
    # 1. Compute centroids (mean vectors)
    dst_sph_mean_vector = cart_centroid(dst)
    src_sph_mean_vector = cart_centroid(src)

    # 2. Compute rotation matrices to align mean vectors to the north pole
    R_dst_sph = rotation_matrix_from_vectors(dst_sph_mean_vector, [0, 0, 1])
    R_src_sph = rotation_matrix_from_vectors(src_sph_mean_vector, [0, 0, 1])

    # 3. Rotate the point clouds
    sph_complete_dst_at_NP = rotate_a_point_cloud(dst, R_dst_sph)
    sph_complete_src_at_NP = rotate_a_point_cloud(src, R_src_sph)

    # 4. Align normals to the z-axis
    dst_at_origin_sphere = align_normalsz(sph_complete_dst_at_NP)
    src_at_origin_sphere = align_normalsz(sph_complete_src_at_NP)
    #dst_at_origin_sphere = sph_complete_dst_at_NP
    #src_at_origin_sphere = sph_complete_src_at_NP

    # 5. Convert to geographic coordinates
    dst_sph_NP_geo = cart2geo_array(dst_at_origin_sphere)
    src_sph_NP_geo = cart2geo_array(src_at_origin_sphere)
    
    ### print maximum
    #print("dst_sph_NP_geo max: ", np.max(dst_sph_NP_geo, axis=0))
    #print("src_sph_NP_geo max: ", np.max(src_sph_NP_geo, axis=0))

    # 6. Generate binary histograms
    dst_hist = binary_hist(hist2D_from_geo_array(dst_sph_NP_geo, lat_bin, long_bin))
    src_hist = binary_hist(hist2D_from_geo_array(src_sph_NP_geo, lat_bin, long_bin))

    # 7. Find the best shift between histograms
    if use_fft:
        shift = find_1D_shift_wfft(flatten_2d(dst_hist), flatten_2d(src_hist))
    else:
        shift = find_1D_shift_wofft(flatten_2d(dst_hist), flatten_2d(src_hist))
    R2_shift = Rz(shift)

    # 8. Compute the final rotation matrix and rotate the source point cloud
    R_comp = np.linalg.inv(R_dst_sph) @ R2_shift @ R_src_sph
    Final_source = rotate_a_point_cloud(src, R_comp)

    return R_comp.T, Final_source


def FRS(dst, src, k=360, max_iterations=50, convergence_angle=0, use_fft=False):
    """
    Estimate the rotation matrix that best aligns a source point cloud to a target point cloud
    using a histogram-based rotational alignment method.

    Args:
    - dst (np.ndarray): An (n, 3) array representing the target point cloud.
    - src (np.ndarray): An (n, 3) array representing the source point cloud that needs alignment.
    - k (int): Number of bins for the histogram (default is 360).
    - max_iterations (int): Maximum number of iterations for the alignment loop (default is 50).
    - convergence_angle (float): The threshold angle (in degrees) for convergence (default is 0).

    Returns:
    - np.ndarray: A (3, 3) rotation matrix that aligns the source point cloud to the target.
    - np.ndarray: The aligned source point cloud.
    """
#def FRS(dst, src, k=360, max_iterations=50, convergence_angle=0, use_fft=False):

    # Convert the target point cloud to rotation angles
    A_normals = pcl_array_to_rotation_angle_array(dst)

    # Initialize the source normals and rotation matrix array
    B_normals = src
    B_normals_array = [B_normals]
    R_comp_arr = []

    # Define the angle range for histograms
    range_min = 0
    range_max = 360

    for i in range(max_iterations):
        # Convert the source point cloud to rotation angles
        B_rot_angles = pcl_array_to_rotation_angle_array(B_normals)

        # Compute histograms for each axis' rotation angles
        A_x_hist = hist1D(A_normals[:, 0], k, range_min, range_max)
        B_x_hist = hist1D(B_rot_angles[:, 0], k, range_min, range_max)

        A_y_hist = hist1D(A_normals[:, 1], k, range_min, range_max)
        B_y_hist = hist1D(B_rot_angles[:, 1], k, range_min, range_max)

        A_z_hist = hist1D(A_normals[:, 2], k, range_min, range_max)
        B_z_hist = hist1D(B_rot_angles[:, 2], k, range_min, range_max)
        
        if use_fft:
            # Find the shifts along each axis
            x_shift = find_1D_shift_wfft(A_x_hist, B_x_hist)
            y_shift = find_1D_shift_wfft(A_y_hist, B_y_hist)
            z_shift = find_1D_shift_wfft(A_z_hist, B_z_hist)
        else:
            # Find the shifts along each axis
            x_shift = find_1D_shift_wofft(A_x_hist, B_x_hist)
            y_shift = find_1D_shift_wofft(A_y_hist, B_y_hist)
            z_shift = find_1D_shift_wofft(A_z_hist, B_z_hist)

        # Compute the rotation matrix for the current shift
        R_comp = rotation_angles2mat(x_shift, y_shift, z_shift)
        R_comp_arr.append(R_comp)

        # Check for convergence
        if abs(x_shift) <= convergence_angle and abs(y_shift) <= convergence_angle and abs(z_shift) <= convergence_angle:
            break

        # Rotate the source point cloud
        B_normals = rotate_a_point_cloud(B_normals, R_comp)
        B_normals_array.append(B_normals)

    # Estimate the final rotation matrix that aligns the original and the final rotated source point clouds
    R_est = estimate_rotation_one_to_one(B_normals_array[0], B_normals_array[-1])

    return R_est, B_normals_array[-1]

def SPMC_FRS(dst, src):
    R_comp_spmc, Final_source_spmc = SPMC(dst, src, lat_bin=180, long_bin=360, use_fft = False)
    R_FRS, final_source = FRS(dst, Final_source_spmc, k=360, max_iterations=50, convergence_angle=0, use_fft=False)
    R_net = R_comp_spmc @ R_FRS
    return R_net, final_source
    
def SPMC_FRS_fft(dst, src):
    R_comp_spmc, Final_source_spmc = SPMC(dst, src, lat_bin=180, long_bin=360, use_fft = True)
    R_FRS, final_source = FRS(dst, Final_source_spmc, k=360, max_iterations=50, convergence_angle=0, use_fft = True)
    R_net = R_comp_spmc @ R_FRS
    return R_net, final_source



########## optinal ##########
def binary_hist(hist):
    """
    Convert a histogram to a binary histogram.

    Args:
        hist (numpy.ndarray): A 2D array representing a histogram.

    Returns:
        numpy.ndarray: A binary 2D histogram where each element is 1 if the corresponding element in the original histogram is greater than 0, otherwise 0.
    """
    return (hist > 0).astype(int)

def align_normalsz(normals, reference=np.array([0, 0, 1])):
    """
    Aligns the normals to the reference direction without modifying the input array in-place.

    Args:
        normals (numpy.ndarray): A numpy array of shape (N, 3) representing the normals.
        reference (numpy.ndarray): A 1D array of length 3 representing the reference direction.
                                   Defaults to [0, 0, 1] (z-axis).

    Returns:
        numpy.ndarray: A numpy array of aligned normals with the same shape as the input.
    """
    # Ensure the reference vector is a unit vector.
    reference = reference / np.linalg.norm(reference)

    # Create a copy of the normals array to avoid in-place modification.
    aligned_normals = normals.copy()
    
    # Calculate the dot product between each normal in the copy and the reference direction.
    dot_product = np.dot(aligned_normals, reference)
    
    # Flip the normals in the copy where the dot product is negative.
    aligned_normals[dot_product < 0] *= -1
    
    return aligned_normals




def find_1D_shift_wfft(gt_hist, moving_hist):
    """
    Find the optimal 1D circular shift between two histograms using normalized cross-correlation via FFT.

    Args:
    - gt_hist (np.ndarray): The reference (ground truth) histogram.
    - moving_hist (np.ndarray): The histogram to be shifted.

    Returns:
    - float: The best shift in degrees that aligns the moving_hist with gt_hist.
    """

    assert len(gt_hist) == len(moving_hist), "Histograms must have the same length."

    k = len(gt_hist)
    n = int(k / 360)  # Conversion factor to map shifts to degrees
    
    # Perform 1D cross-correlation using Fast Fourier Transform
    cc = np.fft.ifft(np.fft.fft(gt_hist) * np.conj(np.fft.fft(moving_hist)))
    
    # Find the index of the maximum cross-correlation and convert to degrees
    best_shift_flat = np.argmax(np.abs(cc)) / n
    
    return best_shift_flat


def convert_to_longitude_range(angle):
    if angle > 180:
        return angle - 360
    else:
        return angle