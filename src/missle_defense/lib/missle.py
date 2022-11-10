
import pygame


class Missle(pygame.sprite.Sprite):
    color = (0, 255, 255)
    score_multi = 1

    def __init__(self, speed = 0.5):
        super(Missle, self).__init__()
        # self.x = 0
        self.center = (0, 0)
        self.surf = pygame.Surface((20, 20))
        self.surf.fill(self.color)
        self.rect = self.surf.get_rect(center=self.center)
        self.speed = speed
        
       

    def set_center(self, x, y):
        self.center = (x, y)
        self.rect = self.surf.get_rect(center=self.center)
        
    def update(self):
        """
        apparently this is called every frame
        and also move_ip needs whole numbers
        
        """
        self.center = (self.center[0] + self.speed, self.center[1])
        # self.rect.move_ip(self.speed,0 )
        # self.set_center(*elf.center)
        self.rect = self.surf.get_rect(center=self.center)
        # print(f"position is {self.rect}")
        
        
        