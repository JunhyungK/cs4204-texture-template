import numpy as np
from transform import Transform


class OrthoCamera:
    def __init__(self, left, right, bottom, top, near, far):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.near = near
        self.far = far
        self.transform = Transform()  

    def ratio(self):
        return (self.right - self.left) / (self.top - self.bottom)
    
    def project_point(self, p):
        # Transform point to camera space using inverse of camera transform
        cam_space_point = self.transform.apply_inverse_to_point(p)

        # Orthographic projection
        ortho_proj = np.array([
            2 * (cam_space_point[0] - self.left) / (self.right - self.left) - 1,
            2 * (cam_space_point[1] - self.bottom) / (self.top - self.bottom) - 1,
            -(2 * (cam_space_point[2] - self.near) / (self.far - self.near) - 1)
        ])
        return ortho_proj
    

    
    def inverse_project_point(self, p):
        # Inverse orthographic projection
        cam_space_point = np.array([
            (p[0] + 1) * (self.right - self.left) / 2 + self.left,
            (p[1] + 1) * (self.top - self.bottom) / 2 + self.bottom,
            (p[2] + 1) * (self.far - self.near) / 2 + self.far
        ])

        # Transform point from camera space to world space
        world_space_point = self.transform.apply_to_point(cam_space_point)
        return world_space_point

class PerspectiveCamera:
    def __init__(self, left, right, bottom, top, near, far):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.near = near
        self.far = far
        self.transform = Transform()

    def ratio(self):
        return (self.right - self.left) / (self.top - self.bottom)
    
    
    def project_point(self, p):
        cam_space_point = self.transform.apply_inverse_to_point(p)
        if cam_space_point[2] == 0:
            raise ValueError("Z coordinate cannot be zero")
        
        perspective_proj = np.array([
            (cam_space_point[0] / cam_space_point[2]) * (self.right - self.left) / 2 + (self.right + self.left) / 2,
            (cam_space_point[1] / cam_space_point[2]) * (self.top - self.bottom) / 2 + (self.top + self.bottom) / 2,
            cam_space_point[2]  
        ])
        perspective_proj[1] = -perspective_proj[1]  # Flip the y-axis for perspective camera
        perspective_proj[0] = -perspective_proj[0]  # Flip the y-axis for perspective camera

        return perspective_proj
    
    def inverse_project_point(self, p):
        z_cam = (self.near * self.far) / ((self.far - self.near) * p[2] + self.near + self.far) * 2

        x_cam = (p[0] * z_cam * (self.right - self.left)) / (2 * self.near)
        y_cam = (p[1] * z_cam * (self.top - self.bottom)) / (2 * self.near) 

        cam_space_point = np.array([x_cam, y_cam, z_cam])

        world_space_point = self.transform.apply_to_point(cam_space_point)

        return world_space_point
