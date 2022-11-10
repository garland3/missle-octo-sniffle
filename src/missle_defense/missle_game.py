# Import the pygame module
import random
import pygame


# Import pygame.locals for easier access to key coordinates
# Updated to conform to flake8 and black standards
# from pygame.locals import *
from pygame.locals import (
    RLEACCEL,
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    K_SPACE,
    KEYDOWN,
    QUIT,
)

from lib.missle import Missle
from lib.defense_gun import DefenseGun

# Define constants for the screen width and height
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480


def run():
    print("run")
    # Set up the drawing window
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
    # Run until the user asks to quit
    
    missles = pygame.sprite.Group()
    all_objects = pygame.sprite.Group()
    def make_missle(speed = 0.5):  
        new_missle = Missle(speed)
        y = random.randint(0, SCREEN_HEIGHT)
        new_missle.set_center(0, y)
        missles.add(new_missle)
        all_objects.add(new_missle)
        return new_missle
    
    missle_speed = 0.5
    missle = make_missle(missle_speed)
    
    ADDMISSLE = pygame.USEREVENT + 1
    rate = 1000
    pygame.time.set_timer(ADDMISSLE, rate)
    
    
    
    defense_gun = DefenseGun(SCREEN_WIDTH, SCREEN_HEIGHT)
    # defense_gun.set_center(SCREEN_WIDTH/2/2, SCREEN_HEIGHT//2)
    
    # all_objects.add(defense_gun)
    
    bullets = pygame.sprite.Group()

    running = True
    cnt = 0
    rotation = 180
    while running:
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == ADDMISSLE:
                missle = make_missle(missle_speed)
            if event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False 
                if event.key == K_SPACE:
                    # Add a new bullet.
                    # when space is pressed, make a bullet
                    bullet = defense_gun.shoot()
                    bullets.add(bullet)
                    all_objects.add(bullet)
                    
                    
        keys = pygame.key.get_pressed()  #checking pressed keys
        # print(keys)
        if keys[pygame.K_LEFT]:
            rotation += 1
            print(f"left {rotation}")
        if keys[pygame.K_RIGHT]:
            rotation-=1
            print(f"right {rotation}")
                
                        
        # Fill the background with white
        screen.fill((255, 255, 255))

        
        # Check if the bullet has collided with anything.
        for b in bullets:
            hit_missle = pygame.sprite.spritecollide(b, missles, True)
            if hit_missle:
                rate -= 10
                missle_speed+=0.05
                print(f"hit!!. New rate is {rate} and new speed is {missle_speed}")
                pygame.time.set_timer(ADDMISSLE, rate)
                
                
        
        all_objects.update()
        for m in all_objects:
            screen.blit(m.surf, m.rect)
            
        # defense_gun.draw_rectangle(SCREEN_WIDTH/2/2, SCREEN_HEIGHT//2, 30, 20, (0, 0, 255), screen, rotation)
        defense_gun.update_gun_angle(screen, rotation)
        # rotation += 2

        # Flip the display
        pygame.display.flip()
        cnt+=1
        if cnt % 100 == 0:
            print(cnt)

        # Ensure program maintains a rate of 30 frames per second
        clock.tick(20)
        
    # Done! Time to quit.
    pygame.quit()


if __name__ == "__main__":
    run()