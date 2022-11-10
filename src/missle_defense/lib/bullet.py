import math
import pygame

class Bullet(pygame.sprite.Sprite):
    color = (0, 0, 255)
    score_multi = 1
    screen_width = 0
    screen_height = 0

    def __init__(self, angle_degrees, speed, position):
        super(Bullet, self).__init__()
        self.center = position
        rot_radians = (math.pi / 180) * angle_degrees
        self.velocity = (math.cos(rot_radians) * speed, math.sin(rot_radians) * speed)
        self.velocity = (self.velocity[0] , self.velocity [1] *-1) # flip y axis
        
        print(f"Now creating bullet at {self.center} with angle {angle_degrees} and speed {speed}. Velocity is {self.velocity}")
        # self.x = 0
        self.surf = pygame.Surface((20, 20))
        self.surf.fill(self.color)
        self.rect = self.surf.get_rect(center=self.center)
        
        
    def update(self):
        """
        apparently this is called every frame
        and also move_ip needs whole numbers
        
        """
        # print(f"curent center is {self.center} before update. Velocity is {self.velocity}")
        # self.center = self.center + self.velocity
        self.center = (self.center[0] + self.velocity[0], self.center[1] + self.velocity[1])
        
        # print(f"curent center is {self.center} after")
        
        # self.center

        self.rect = self.surf.get_rect(center=self.center)
        # print(f"Bullet position is {self.rect}")
        if self.center[0] > self.screen_width or self.center[0] < 0 or self.center[1] > self.screen_height or self.center[1] < 0:
            self.kill()
        