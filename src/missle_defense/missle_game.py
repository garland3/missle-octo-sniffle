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

from missle_defense.lib.missle import Missle
from missle_defense.lib.defense_gun import DefenseGun

# Define constants for the screen width and height
# SCREEN_WIDTH = 640
# SCREEN_HEIGHT = 480

SCREEN_WIDTH = 300
SCREEN_HEIGHT = 300

pygame.init()

class MissleGame:
    def __init__(self, gym_env =False, show_screen=True):
        print("starting game")
        # Set up the drawing window
        self.clock = pygame.time.Clock()
        self.gym_env = gym_env
        # print(f"gym_env Bool: {gym_env}")

        self.show_screen = show_screen
        # if self.show_screen is True:
        self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        # Run until the user asks to quit

        self.missles = pygame.sprite.Group()
        self.all_objects = pygame.sprite.Group()


        self.missle_speed = 2.0
        self.ADDMISSLE = pygame.USEREVENT + 1
        self.rate = 50
        # pygame.time.set_timer(self.ADDMISSLE, self.rate)
        # Create a custom event for adding more bullets to the ammo box.        
        # self.ADD_BULLET = pygame.USEREVENT + 2
        # self.rate_new_bullet = 500
        # pygame.time.set_timer(self.ADD_BULLET, self.rate_new_bullet)
        self.bullets_cnt = 0

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
    
    def process_shoot(self):
        # Add a new bullet.
        # when space is pressed, make a bullet
        if self.bullets_cnt>0:
            bullet = self.defense_gun.shoot()
            self.bullets.add(bullet)
            self.all_objects.add(bullet)
            self.bullets_cnt -= 1
            # print("bullet cnt: ", self.bullets_cnt)
        # else:
            # print("no more bullets")
        

    def step(self, action=None):
        """
        step the game forward one frame
        """
        # print(f"action: {action} and cnt: {self.cnt} and env: {self.gym_env}")
        
        # Did the user click the window close button?
        if self.cnt % 20 == 0:
            if self.bullets_cnt < 100:
                self.bullets_cnt += 1
        
        if self.cnt % self.rate == 0:
            _ = self.make_missle(self.missle_speed)
            
        if self.gym_env is False:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                # if event.type == self.ADDMISSLE:
                if event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
                    if event.key == K_SPACE:
                        self.process_shoot()

            keys = pygame.key.get_pressed()  # checking pressed keys
            # print(keys)
            if keys[pygame.K_LEFT]:
                self.rotation += 3
                # print(f"left {rotation}")
            if keys[pygame.K_RIGHT]:
                self.rotation -= 3
                # print(f"right {rotation}")
        
        # ---------------------------------
        # PROCESS the gym setup. 
        # ---------------------------------        
        if self.gym_env is True:
            # print(f"action: {action}")
            # For the gym, we need to provide the action
            if action is not None:
                if action == 0:
                    self.rotation += 3
                if action == 1:
                    self.rotation -= 3
                if action == 2:
                    self.process_shoot()
        
        # wrap the rotation
        if self.rotation > 360:
            self.rotation = self.rotation - 360            

        # Fill the background with white
        # if self.show_screen is True:
        self.screen.fill((255, 255, 255))

        # Check if the bullet has collided with anything.
        for b in self.bullets:
            hit_missle = pygame.sprite.spritecollide(b, self.missles, True)
            if hit_missle:
                b.kill()
                self.rate -= 10
                if self.rate<2:
                    self.rate = 2
                self.missle_speed += 0.05
                print(f"hit!!. New self.rate is {self.rate} and new speed is {self.missle_speed}, score is {self.score}")
                pygame.time.set_timer(self.ADDMISSLE, self.rate)
                self.score += 100

        for m in self.missles:
            if m.rect.x > SCREEN_WIDTH:
                # print("missle past screen")
                print(f"Final score is {self.score}")
                self.running = False
                return

        self.all_objects.update()
        # if self.show_screen is True:
        for m in self.all_objects:
            self.screen.blit(m.surf, m.rect)

        # defense_gun.draw_rectangle(SCREEN_WIDTH/2/2, SCREEN_HEIGHT//2, 30, 20, (0, 0, 255), screen, rotation)
        self.defense_gun.update_gun_angle(self.rotation)
        # if self.show_screen is True:
        self.defense_gun.draw_gun(self.screen)
            
        # Flip the display
        if self.show_screen is True:
            pygame.display.flip()
        self.cnt += 1
        # if self.cnt % 100 == 0:
        #     print(self.cnt)
        
    # def end_the_game(self):
    #     # pygame.display.quit()
        

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
        # self.end_the_game()
        pygame.quit()
        print("game ended")
        return
        


if __name__ == "__main__":
    mg = MissleGame()
    mg.run()
