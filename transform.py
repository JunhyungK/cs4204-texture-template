import numpy as np

class Transform:
    def __init__(self):
        self.my_transformation_matrix = np.identity(4)
        self._rotation_matrix = np.identity(4)
        self._position_matrix = np.identity(4)



    # Returns a 4x4 Numpy matrix that represents the transformation matrix
    def transformation_matrix(self):
        return self.my_transformation_matrix

    # This method takes three scalars (x, y, and z) as input, and updates the Transform object's internal state to represent a new position at (x, y, z).
    def set_position(self, x, y, z):
        self._position_matrix = np.identity(4)
        self._position_matrix[0, 3] = x
        self._position_matrix[1, 3] = y
        self._position_matrix[2, 3] = z
        self._update_transformation_matrix()

    def convert_deg_rad(self, degrees):
        return degrees * np.pi / 180
    
    # This method takes three scalars (x, y, and z) as input, and updates the Transform object's interal rotation state. 
    # The input values x, y, and z are expected to be degrees values between 0.0 and 360.0 (there is no need to check this) , 
    # and represent the amount of rotation around the x, y, and z axis respectively. 
    # The rotation should be set using the ZYX order of rotation.
    def set_rotation(self, x, y, z):
        # Convert degrees to radians
        rad_x = self.convert_deg_rad(x)
        rad_y = self.convert_deg_rad(y)
        rad_z = self.convert_deg_rad(z)

        # Apply transformation
        rotation_z = np.array([
            [np.cos(rad_z), -np.sin(rad_z), 0, 0],
            [np.sin(rad_z), np.cos(rad_z),  0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Rotation matrix around Y axis (pitch)
        rotation_y = np.array([
            [np.cos(rad_y),  0, np.sin(rad_y), 0],
            [0, 1, 0, 0],
            [-np.sin(rad_y), 0, np.cos(rad_y), 0],
            [0, 0, 0, 1]
        ])

        rotation_x = np.array ([
            [1, 0, 0, 0],
            [0, np.cos(rad_x), -np.sin(rad_x), 0],
            [0, np.sin(rad_x), np.cos(rad_x), 0],
            [0, 0, 0, 1]
        ])
    
        self._rotation_matrix = rotation_x @ rotation_y @ rotation_z
        self._update_transformation_matrix()

    
    # This method returns a 4x4 Numpy matrix that is the inverse of the transformation matrix.
    def inverse_matrix(self):
        inv =  np.linalg.inv(self.my_transformation_matrix)
        return inv


    # This method takes a 3 element Numpy array, p, that represents a 3D point in space as input. 
    # It then applies the transformation matrix to it, and returns the resulting 3 element Numpy array.
    def apply_to_point(self, p):
        p = np.array(p)
        if p.ndim == 2 and p.shape[0] == 1:
            p = p.flatten()  

        p_homogeneous = np.array([*p, 1])
        transformed_point = self.my_transformation_matrix @ p_homogeneous 
        return transformed_point[:3] 


    # This method takes a 3 element Numpy array, p, that represents a 3D point in space as input. 
    # It then applies the inverse transformation matrix to it, and returns the resulting 3 element Numpy array.
    def apply_inverse_to_point(self, p):
        if p.ndim == 2 and p.shape[0] == 1:
            p = p.flatten() 

        return self.inverse_matrix()[:3] @ np.array([*p, 1])


    # This method takes a 3 element Numpy array, n, that represents a 3D normal vector. 
    # It then applies the transform's rotation to it, and returns the resulting 3 element Numpy array. 
    # The resulting array should be normalized and should not be affected by any positional component within the transform.
    def apply_to_normal(self, n):

        # Create a rotation matrix without translation
        rotation_matrix = self._rotation_matrix[:3, :3]  # Get the 3x3 rotation part

        # Apply the rotation to the normal
        transformed_normal = rotation_matrix @ n

        # Normalize the resulting normal vector
        norm = np.linalg.norm(transformed_normal)

    
        return transformed_normal / norm  # Return the normalized normal vector
    

    # Based on the wikipedia explanation
    def set_axis_rotation(self, axis, rotation):
          # Normalize the axis vector
        axis = axis / np.linalg.norm(axis)
        rad_rotation = np.radians(rotation)

        # Create the skew-symmetric matrix K
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        # Calculate the rotation matrix using Rodrigues' rotation formula
        I = np.identity(3)
        R = I + np.sin(rad_rotation) * K + (1 - np.cos(rad_rotation)) * np.dot(K, K)

        # Update the rotation matrix to be a 4x4 matrix
        self._rotation_matrix[0:3, 0:3] = R

        # Update the transformation matrix
        self._update_transformation_matrix()

    def _update_transformation_matrix(self):
        # Update the transformation matrix by combining position and rotation
        self.my_transformation_matrix = self._position_matrix @ self._rotation_matrix
