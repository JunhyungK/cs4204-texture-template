import math
import numpy as np
from transform import Transform
from stl import mesh
from PIL import Image

np.set_printoptions(precision=16, suppress=False)

class Mesh:
    def __init__(self, diffuse_color, specular_color, ka, kd, ks, ke):
        """
        Initializes the mesh with material properties and prepares lists for vertices, faces, and normals.
        
        Parameters:
            diffuse_color (array-like): RGB values for diffuse color (clipped between 0.0 and 1.0).
            specular_color (array-like): RGB values for specular color (clipped between 0.0 and 1.0).
            ka (float): Ambient reflection coefficient.
            kd (float): Diffuse reflection coefficient.
            ks (float): Specular reflection coefficient.
            ke (float): Emission coefficient.
        """
        # Initialize mesh properties
        self.verts = []
        self.faces = []
        self.normals = []
        self.vertex_map = {}
        self.transform = Transform()

        # Store material properties with clipping to ensure they are within valid range [0, 1]
        self.diffuse_color = np.clip(np.array(diffuse_color), 0.0, 1.0)
        self.specular_color = np.clip(np.array(specular_color), 0.0, 1.0)
        self.ka = ka  # Ambient reflection coefficient
        self.kd = kd  # Diffuse reflection coefficient
        self.ks = ks  # Specular reflection coefficient
        self.ke = ke  # Emission coefficient

    def add_vertex(self, vertex):
        """
        Adds a vertex to the mesh, ensuring no duplicates by using a vertex map.
        Returns the index of the vertex in the mesh.
        
        Parameters:
            vertex (array-like): A 3D vertex (x, y, z) to add to the mesh.
        
        Returns:
            int: Index of the added vertex in the mesh.
        """
        vertex = list(vertex)  # Convert vertex to list for immutability compatibility
        vertex_tuple = tuple(vertex)  # Convert to tuple for hashing in the map

        # Check if the vertex is already in the vertex map
        if vertex_tuple in self.vertex_map:
            return self.vertex_map[vertex_tuple]
        
        # Add new vertex if not already present
        self.verts.append(vertex)
        index = len(self.verts) - 1
        self.vertex_map[vertex_tuple] = index
        return index

    def calculate_normal(self, v1, v2, v3):
        """
        Calculates the normal vector for a triangle defined by three vertices.
        
        Parameters:
            v1, v2, v3 (array-like): 3D coordinates of the vertices of the triangle.
        
        Returns:
            np.ndarray: Normalized normal vector of the triangle.
        """
        edge1 = np.array(v2) - np.array(v1)
        edge2 = np.array(v3) - np.array(v1)
        normal = np.cross(edge1, edge2)
        return normal / np.linalg.norm(normal)  # Normalize the normal vector

    def process_face(self, face):
        my_face = [self.add_vertex(vertex) for vertex in face]
        self.faces.append(my_face)

        # Calculate and store the normal of the face
        normal = self.calculate_normal(face[0], face[1], face[2])
        self.normals.append(normal)


    def vertex_normals (self):
        vertex_normal_accumulator = np.zeros((len(self.verts), 3))

        for face, face_normal in zip(self.faces, self.normals):
            for vertex_idx in face:
                vertex_normal_accumulator[vertex_idx] += face_normal

        vertex_normals = np.array([
            normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else normal   
            for normal in vertex_normal_accumulator
        ])

        return vertex_normals
    
    def planar_uvs(self, axis='z'):
        """
        Generates UV coordinates for the vertices of the mesh using planar projection.
        
        Parameters:
            axis (str): The axis to project the UV coordinates onto ('x', 'y', or 'z').
        """
        self.uvs = []
        
        for vertex in self.verts:
            x, y, z = vertex
            if axis == 'x':
                u, v = y, z
            elif axis == 'y':
                u, v = x, z
            else:  # default to 'z' axis
                u, v = x, y
            
            # Normalize u and v to be between 0 and 1
            u = (u - np.min(self.verts, axis=0)[0]) / (np.max(self.verts, axis=0)[0] - np.min(self.verts, axis=0)[0])
            v = (v - np.min(self.verts, axis=0)[1]) / (np.max(self.verts, axis=0)[1] - np.min(self.verts, axis=0)[1])
            
            self.uvs.append([u, v])
    def load_texture(self, texture_path):
        """
        Loads the texture image and stores it.
        """
        self.texture = Image.open(texture_path)
        self.texture = np.array(self.texture)
        self.texture_height, self.texture_width, _ = self.texture.shape


    def camera_space_uv_mapping(self, camera):
        # Generate texture coordinates based on camera space
        uv_coords = []
        for vertex in self.verts:
            # Transform the vertex to camera space (from world space)
            camera_space_vertex = camera.transform.apply_to_point(vertex)

            # Compute "UV" coordinates based on camera space
            u = 0.5 + np.arctan2(camera_space_vertex[2], camera_space_vertex[0]) / (2 * np.pi)
            v = 0.5 - np.arcsin(camera_space_vertex[1] / np.linalg.norm(camera_space_vertex)) / np.pi
            uv_coords.append((u, v))
        return uv_coords

    @classmethod
    def from_stl(cls, stl_path, diffuse_color, specular_color, ka, kd, ks, ke):
        """
        Loads a mesh from an STL file and returns a Mesh object populated with vertices, faces, and normals.
        
        Parameters:
            stl_path (str): Path to the STL file.
            diffuse_color (array-like): RGB values for diffuse color.
            specular_color (array-like): RGB values for specular color.
            ka (float): Ambient reflection coefficient.
            kd (float): Diffuse reflection coefficient.
            ks (float): Specular reflection coefficient.
            ke (float): Emission coefficient.
        
        Returns:
            Mesh: A populated Mesh object.
        """

        # Create a new mesh object
        mesh_obj = cls(diffuse_color, specular_color, ka, kd, ks, ke)

        # Load the STL file using the 'stl' library
        stl_mesh_obj = mesh.Mesh.from_file(stl_path)

        # Process each face in the STL file
        for face in stl_mesh_obj.vectors:
            mesh_obj.process_face(face)

        return mesh_obj
