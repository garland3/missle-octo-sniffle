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


class MissleGame:
    def __init__(self, show_screen=True):
        print("run")
        # Set up the drawing window
        self.clock = pygame.time.Clock()

        self.show_screen = show_screen
        if self.show_screen is True:
            self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        # Run until the user asks to quit

        self.missles = pygame.sprite.Group()
        self.all_objects = pygame.sprite.Group()


        self.missle_speed = 0.5

        self.ADDMISSLE = pygame.USEREVENT + 1
        self.rate = 1000
        pygame.time.set_timer(self.ADDMISSLE, self.rate)

        self.defense_gun = DefenseGun(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.bullets = pygame.sprite.Group()
        self.score = 0
        self.running = True
        self.cnt = 0
        self.rotation = 180
        
    def make_missle(self,speed=0.5):
        new_missle = Missle(speed)
        y = random.randint(0, SCREEN_HEIGHT)
        new_missle.set_center(0, y)
        self.missles.add(new_missle)
        self.all_objects.add(new_missle)
        return new_missle

    def step(self):
        """
        step the game forward one frame
        """
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == self.ADDMISSLE:
                _ = self.make_missle(self.missle_speed)
            if event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                if event.key == K_SPACE:
                    # Add a new bullet.
                    # when space is pressed, make a bullet
                    bullet = self.defense_gun.shoot()
                    self.bullets.add(bullet)
                    self.all_objects.add(bullet)

        keys = pygame.key.get_pressed()  # checking pressed keys
        # print(keys)
        if keys[pygame.K_LEFT]:
            self.rotation += 1
            # print(f"left {rotation}")
        if keys[pygame.K_RIGHT]:
            self.rotation -= 1
            # print(f"right {rotation}")

        # Fill the background with white
        if self.show_screen is True:
            self.screen.fill((255, 255, 255))

        # Check if the bullet has collided with anything.
        for b in self.bullets:
            hit_missle = pygame.sprite.spritecollide(b, self.missles, True)
            if hit_missle:
                b.kill()
                self.rate -= 10
                self.missle_speed += 0.05
                print(f"hit!!. New self.rate is {self.rate} and new speed is {self.missle_speed}, score is {self.score}")
                pygame.time.set_timer(self.ADDMISSLE, self.rate)
                self.score += 100

        for m in self.missles:
            if m.rect.x > SCREEN_WIDTH:
                print("missle past screen")
                print(f"score is {self.score}")
                self.running = False
                return

        self.all_objects.update()
        if self.show_screen is True:
            for m in self.all_objects:
                self.screen.blit(m.surf, m.rect)

        # defense_gun.draw_rectangle(SCREEN_WIDTH/2/2, SCREEN_HEIGHT//2, 30, 20, (0, 0, 255), screen, rotation)
        self.defense_gun.update_gun_angle(self.rotation)
        if self.show_screen is True:
            self.defense_gun.draw_gun(self.screen)
            
        # rotation += 2

        # Flip the display
        pygame.display.flip()
        self.cnt += 1
        if self.cnt % 100 == 0:
            print(self.cnt)

    def run(self):
        """
        Just run the game
        """
       
        while self.running:
            self.step()
            if self.running == False:
                break
            # Ensure program maintains a rate of 30 frames per second
            self.clock.tick(20)

        # Done! Time to quit.
        pygame.quit()


if __name__ == "__main__":
    mg = MissleGame()
    mg.run()
