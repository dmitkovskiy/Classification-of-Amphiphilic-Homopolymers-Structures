import numpy as np

BATCH_SIZE = 60
NUM_CLASSES = 6
N_SHIFT = 10

def rotate_to_principal_axes(points: np.ndarray, min_axes: str = 'Z'):
    
    points_copy = points[:, :3]
    center = points_copy.mean(0)

    normed_points = points_copy - center[None,:]
    gyr = np.einsum('im,in->mn', normed_points,normed_points)/len(points_copy)
    w, v = np.linalg.eig(gyr)

    rot_points = np.einsum('ij,jk',normed_points,v)
    rot_center = np.einsum('ij,jk',np.array(center)[np.newaxis, :],v)
    rot_points = rot_points + rot_center
    
    idx = w.argsort()
    if min_axes == 'Z':
        rot_points = rot_points[:, idx[::-1]]
    elif min_axes == 'X':
        rot_points = rot_points[:, idx]
    else:
        print('Минимальная ось указана неверно, ничего не делаю')

    if len(points[0])>3:
        types_column = points[:, 3]  # Четвертый столбец (types)
        rot_points = np.column_stack((rot_points, types_column))

    return rot_points

def drop_points_a_da(points: np.ndarray, a: float, da: float) -> np.ndarray:
    
    distances = np.linalg.norm(points[:, :3], axis=1)
    mask = np.logical_and(distances >= a, distances <= a + da)
    filtered_points = points[mask]
    
    if len(filtered_points) == 0:
        print('Не существует точек в указанной области')

    return filtered_points

def cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(np.sum(points[:, :3]**2, axis=1))
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    spherical_points = np.column_stack((r, theta, phi))

    if len(points[0])>3:
        spherical_points = np.column_stack((spherical_points, points[:, 3] ))
    
    return spherical_points

def spherical_to_cartesian(points):

    r, theta, phi = points[:, 0], points[:, 1], points[:, 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    cartesian_points = np.column_stack((x, y, z))

    if len(points[0])>3:
        cartesian_points = np.column_stack((cartesian_points, points[:, 3]))

    return cartesian_points

def flatten_points(points, a):
    spherical_points = cartesian_to_spherical(points)
    spherical_points[:, 0] = a
    cartesian_coords = spherical_to_cartesian(spherical_points)

    return cartesian_coords

def rotate_drop_flatten(points, a= 4, da= 6):
    r_points = rotate_to_principal_axes(points)
    d_r_points = drop_points_a_da(r_points, a, da)
    if len(d_r_points)>0:
        f_d_r_points = flatten_points(d_r_points, a)
        return f_d_r_points
    else:
        return []
