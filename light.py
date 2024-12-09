import numpy as np
from transform import Transform

class PointLight:
    def __init__(self, intensity, color):
        self.transform = Transform()
        self.intensity = intensity
        self.color = np.array(color)
        
    def get_position(self):
        # Get the position of the light from its transformation matrix
        return self.transform.transformation_matrix()[:3, 3] 