import math
import pygame

from missle_defense.lib.bullet import Bullet

class DefenseGun:
    def __init__(self, screen_width, screen_height) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.gun_width = 70
        self.gun_height = 10
        self.angle = 0
        

    def draw_rectangle(self,x, y, width, height, color, screen, rotation=0):
        """Draw a rectangle, centered at x, y.

        Arguments:
        x (int/float):
            The x coordinate of the center of the shape.
        y (int/float):
            The y coordinate of the center of the shape.
        width (int/float):
            The width of the rectangle.
        height (int/float):
            The height of the rectangle.
        color (str):
            Name of the fill color, in HTML format.
        """
        points = []

        # The distance from the center of the rectangle to
        # one of the corners is the same for each corner.
        radius = math.sqrt((height / 2)**2 + (width / 2)**2)

        # Get the angle to one of the corners with respect
        # to the x-axis.
        angle = math.atan2(height / 2, width / 2)

        # Transform that angle to reach each corner of the rectangle.
        angles = [angle, -angle + math.pi, angle + math.pi, -angle]

        # Convert rotation from degrees to radians.
        rot_radians = (math.pi / 180) * rotation

        # Calculate the coordinates of each point.
        for angle in angles:
            y_offset = -1 * radius * math.sin(angle + rot_radians)
            x_offset = radius * math.cos(angle + rot_radians)
            points.append((x + x_offset, y + y_offset))

        pygame.draw.polygon(screen, color, points)
        
    def update_gun_angle(self, rotation):
        """
        Update the gun based on some rotation. 
        Gun will be drawn at the bottom of the screen
        """
        self.angle = rotation
        
    def draw_gun(self, screen):
        """
        Draw the gun at the bottom of the screen
        use the angle to rotate the gun
        """
        
        self.draw_rectangle(self.screen_width//2, self.screen_height-10, self.gun_width, self.gun_height, (0, 0, 255), screen, self.angle)

    def shoot(self):
        # print("shoot")
        bullet = Bullet(self.angle, 10, (self.screen_width//2, self.screen_height-10))
        bullet.screen_width = self.screen_width
        bullet.screen_height = self.screen_height
        return bullet
        # make a bullet
    
    # def update(self):
    #     degrees = 10
    #     old = self.
    #     rotated = pygame.transform.rotate(self.surf, degrees)
    #     self.surf = rotated
    #     self.rect = self.surf.get_rect()
        
        