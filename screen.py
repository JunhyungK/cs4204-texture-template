import numpy as np
import pygame

class Screen:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        pygame.init()
        self.screen = pygame.display.set_mode([width, height])

    def ratio(self):
        return self.width / self.height
    
    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def draw(self, buf: np.ndarray):
        """Takes a buffer of 8-bit RGB pixels and puts them on the canvas.
        buf should be a ndarray of shape (height, width, 3)"""
        if buf.shape != (self.height, self.width, 3):
            raise Exception("buffer and screen not the same size")

        # Flip buffer to account for coordinate system differences
        buf = np.flipud(buf)  # Correctly flip vertically

        # Draw the buffer to the screen
        pygame.pixelcopy.array_to_surface(self.screen, buf)

        # Update the display
        pygame.display.flip()

    def draw_line(self, a, b, color, buf: np.ndarray):
        """Implement line drawing from point(s) a to b."""
        for (x0, y0), (x1, y1) in zip(a, b):
            # Calculate the delta x and delta y
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)

            # Determine where to move 
            move_x = 1 if x0 < x1 else -1
            move_y = 1 if y0 < y1 else -1

            err = dx - dy
            
            while True:
                if 0 <= x0 < self.width and 0 <= y0 < self.height:
                    buf[y0, x0] = color  # Access as (row, column)
                
                if x0 == x1 and y0 == y1:
                    break
                
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += move_x
                if e2 < dx:
                    err += dx
                    y0 += move_y
        self.draw(buf)

    def draw_polygon(self, points, color, buf: np.ndarray):
        """Implement line drawing from point(s) a to b, then color the pixels within the defined polygon."""
        # This method needs to be implemented
        pass

    def show(self):
        """Shows the canvas."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        from datetime import datetime
                        pygame.image.save(self.screen, datetime.now().strftime("./%m-%d-%Y_%H--%M--%S.png"))
        pygame.quit()

    def device_to_screen(self, p):
        x = int((p[0] + 1) * self.width / 2)
        y = int((1 - (p[1] + 1) * self.height / 2) * -1)
        return x, y 
