import numpy as np
from screen import Screen
from mesh import Mesh
from transform import Transform
from camera import OrthoCamera, PerspectiveCamera
from light import PointLight
np.set_printoptions(precision=16, suppress=True)
class Renderer:
    def __init__(self, screen, camera, meshes, light):
        self.screen = screen
        self.camera = camera
        self.meshes = meshes
        self.light = light

    def render(self, shading, bg_color, ambient_light):
        height, width = self.screen.get_height(), self.screen.get_width()

        # Initialize image and depth buffers
        image_buffer = np.full((height, width, 3), bg_color, dtype=np.uint8)
        depth_buffer = np.full((height, width), -np.inf, dtype=np.float32)

        for mesh in self.meshes:
            # Transform vertices into camera space and project to screen space
            verts_camera_coords = [
                self.camera.project_point(mesh.transform.apply_to_point(v))
                for v in mesh.verts
            ]
            
            verts_screen_coords = [
                ((-v[1] + 1) / 2 * width, ((-v[0] + 1) / 2) * height)
                for v in verts_camera_coords
            ]
            
            for face, normal in zip(mesh.faces, mesh.normals):
                # Transform face normal into camera space
                transformation_matrix = mesh.transform.transformation_matrix()
                transformed_normal = np.linalg.inv(transformation_matrix[:3, :3]).T.dot(normal)
                transformed_normal = transformed_normal / np.linalg.norm(transformed_normal)

                # View direction
                view_direction = np.array([0, 0, -1])

                # Back-face culling
                if np.dot(transformed_normal, view_direction) >= 0:
                    continue

                # Extract triangle vertices and project
                triangle_verts = [verts_screen_coords[i] for i in face]
                triangle_camera_verts = [verts_camera_coords[i] for i in face]
                triangle_normals = [mesh.vertex_normals()[i] for i in face]

                # Compute triangle bounding box
                min_x = max(0, int(min(x for x, y in triangle_verts)))
                max_x = min(width - 1, int(max(x for x, y in triangle_verts)))
                min_y = max(0, int(min(y for x, y in triangle_verts)))
                max_y = min(height - 1, int(max(y for x, y in triangle_verts)))
                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        if self.is_point_in_triangle((x, y), triangle_verts):
                            depth = self.interpolate_depth((x, y), triangle_verts, triangle_camera_verts)

                            if depth > depth_buffer[y, x]:
                                depth_buffer[y, x] = depth

                                # Calculate screen space coordinates from camera space
                                screen_u, screen_v = self.camera_space_to_screen_space((x, y), triangle_verts, triangle_camera_verts)

                                # Sample texture color based on the transformed screen coordinates
                                texture_color = self.sample_texture(screen_u, screen_v, mesh.texture)

                                if shading == "flat":
                                    color = self.calculate_flat(transformed_normal, ambient_light, mesh)
                                elif shading == "barycentric":
                                    color = self.calculate_bary((x, y), triangle_verts, mesh)
                                elif shading == "phong":
                                    color = self.calculate_phong((x, y), triangle_verts, triangle_normals, face, mesh, ambient_light)
                                else:
                                    color = bg_color

                                # Apply texture color
                                image_buffer[y, x] = np.clip(texture_color, 0, 255)

        self.screen.draw(image_buffer)

    def camera_space_to_screen_space(self, pt, triangle_verts, triangle_camera_verts):
        # Barycentric interpolation for camera-space mapping to screen space
        x, y = pt
        p0, p1, p2 = triangle_verts
        c0, c1, c2 = triangle_camera_verts

        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        w1 = ((p1[1] - p2[1]) * (x - p2[0]) + (p2[0] - p1[0]) * (y - p2[1])) / denom
        w2 = ((p2[1] - p0[1]) * (x - p2[0]) + (p0[0] - p2[0]) * (y - p2[1])) / denom
        w3 = 1 - w1 - w2

        # Interpolate camera space x, y coordinates
        camera_x = w1 * c0[0] + w2 * c1[0] + w3 * c2[0]
        camera_y = w1 * c0[1] + w2 * c1[1] + w3 * c2[1]

        # Project the camera-space coordinates to screen space
        u = (camera_x + 1) / 2  
        v = (camera_y + 1) / 2  

        return u, v

    def sample_texture(self, u, v, texture):
        # Sample the texture using the interpolated camera space coordinates
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        
        # Convert the texture coordinates to pixel coordinates
        height, width, _ = texture.shape
        x = int(u * width)
        y = int(v * height)

        return texture[y, x]


    def calculate_flat(self, normal, ambient_light, mesh):
        light_position = self.light.transform.transformation_matrix()[:3, 3]
        light_dir = light_position - mesh.transform.transformation_matrix()[:3, 3]
        light_dir /= np.linalg.norm(light_dir)

        diffuse_intensity = max(np.dot(normal, light_dir), 0) / np.pi
        ambient = np.array(ambient_light)
        diffuse_color = mesh.diffuse_color * diffuse_intensity

        color = ambient * mesh.ka + diffuse_color * mesh.kd
        return (color * 255).astype(np.uint8)

    def calculate_phong(self, pt, triangle_verts, triangle_normals, face, mesh, ambient_light):
        x, y = pt
        p0, p1, p2 = triangle_verts
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        w1 = ((p1[1] - p2[1]) * (x - p2[0]) + (p2[0] - p1[0]) * (y - p2[1])) / denom
        w2 = ((p2[1] - p0[1]) * (x - p2[0]) + (p0[0] - p2[0]) * (y - p2[1])) / denom
        w3 = 1 - w1 - w2

        # Interpolate the UV coordinates
        uvs = [mesh.uvs[i] for i in [face[0], face[1], face[2]]]
        u = w1 * uvs[0][0] + w2 * uvs[1][0] + w3 * uvs[2][0]
        v = w1 * uvs[0][1] + w2 * uvs[1][1] + w3 * uvs[2][1]

        # Map the UV coordinates to the texture
        texture_x = int(u * mesh.texture_width)
        texture_y = int(v * mesh.texture_height)
        
        # Ensure the coordinates are within bounds
        texture_x = np.clip(texture_x, 0, mesh.texture_width - 1)
        texture_y = np.clip(texture_y, 0, mesh.texture_height - 1)

        # Get the color from the texture
        texture_color = mesh.texture[texture_y, texture_x]

        # Proceed with shading calculations as usual
        # (diffuse, specular, ambient light calculations, etc.)
        # Use texture_color instead of interpolated colors for the final result
        return texture_color
 
 
    def calculate_bary(self, pt, triangle_verts, mesh):
        x, y = pt
        p0, p1, p2 = triangle_verts
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        w1 = ((p1[1] - p2[1]) * (x - p2[0]) + (p2[0] - p1[0]) * (y - p2[1])) / denom
        w2 = ((p2[1] - p0[1]) * (x - p2[0]) + (p0[0] - p2[0]) * (y - p2[1])) / denom
        w3 = 1 - w1 - w2

        color0 = np.array([1.0, 0.0, 0.0])
        color1 = np.array([0.0, 1.0, 0.0])
        color2 = np.array([0.0, 0.0, 1.0])

        interpolated_color = w1 * color0 + w2 * color1 + w3 * color2
        return (interpolated_color * 255).astype(np.uint8)

    def interpolate_depth(self, pt, triangle_verts, triangle_camera_verts):
        x, y = pt
        p0, p1, p2 = triangle_verts
        c0, c1, c2 = triangle_camera_verts

        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        w1 = ((p1[1] - p2[1]) * (x - p2[0]) + (p2[0] - p1[0]) * (y - p2[1])) / denom
        w2 = ((p2[1] - p0[1]) * (x - p2[0]) + (p0[0] - p2[0]) * (y - p2[1])) / denom
        w3 = 1 - w1 - w2

        depth = w1 * c0[2] + w2 * c1[2] + w3 * c2[2]
        return depth

    def is_point_in_triangle(self, pt, verts):
        x, y = pt
        v0, v1, v2 = np.array(verts[0]), np.array(verts[1]), np.array(verts[2])

        # Vectors from the point to the vertices
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = np.array([x, y]) - v0

        # Barycentric coordinates calculation
        dot00 = np.dot(v0v1, v0v1)
        dot01 = np.dot(v0v1, v0v2)
        dot02 = np.dot(v0v1, v0p)
        dot11 = np.dot(v0v2, v0v2)
        dot12 = np.dot(v0v2, v0p)

        # Compute the inverse of the determinant (denom)
        denom = dot00 * dot11 - dot01 * dot01

        # If denom is too small, return False (degenerate triangle)
        if np.abs(denom) < 1e-8:  # Small threshold to handle degenerate cases
            return False

        # Compute barycentric coordinates
        invDenom = 1 / denom
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        # Check if the point is inside the triangle
        return (u >= 0) and (v >= 0) and (u + v <= 1)
