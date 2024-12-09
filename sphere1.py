import numpy as np
from screen import Screen
from camera import PerspectiveCamera
from mesh import Mesh
from renderer import Renderer
from light import PointLight
import pygame


def setup_scene(screen):
    """Sets up the camera, mesh, and light for rendering."""
    camera = PerspectiveCamera(-1.0, 1.0, -1.0, 1.0, -1.0, -10)
    camera.transform.set_position(0, 0, 2.5)

    mesh = Mesh.from_stl("STL_files/suzanne.stl", np.array([1.0, 0.0, 1.0]),
                         np.array([1.0, 1.0, 1.0]), 0.05, 1.0, 0.5, 1000)
    mesh.planar_uvs()
    mesh.load_texture("Tethered.jpg")

    light = PointLight(50.0, np.array([1, 1, 1]))
    light.transform.set_position(0, 5, 5)

    renderer = Renderer(screen, camera, [mesh], light)

    return camera, mesh, light, renderer


def render_view(renderer, mesh, rotation, filename):
    """Renders the mesh with the specified rotation."""
    mesh.transform.set_rotation(*rotation)
    renderer.render("phong", [80, 80, 80], [0.2, 0.2, 0.2])
    pygame.image.save(renderer.screen.screen, filename)
    print(f"Rendered {filename} with rotation {rotation}")


if __name__ == '__main__':
    screen = Screen(500, 500)

    # Setup the scene
    camera, mesh, light, renderer = setup_scene(screen)

    # Render views with increasing y-axis rotation
    for i in range(36):
        rotation = (15, i * 10, 0)
        filename = f"view{i+1}.png"
        render_view(renderer, mesh, rotation, filename)

    print("Rendering complete. Images saved as view1.png to view14.png.")

    # Show the last view on the screen
    screen.show()
