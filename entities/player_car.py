import pygame
from entities.vehicle import Vehicle


class PlayerCar(Vehicle):
    """Player-controlled vehicle using keyboard input (WASD)."""

    def __init__(self, x, y):
        super().__init__(x, y)
        self._color = (0, 120, 255)
    
    def handle_input(self):
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.accelerate(self.ACCELERATION)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.accelerate(-self.ACCELERATION * 0.5)

        # Steering - reversed when reversing (like real car)
        if abs(self.speed) > 0.5:
            direction = 1 if self.speed > 0 else -1
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.rotate(-self.ROTATION_SPEED * direction)
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.rotate(self.ROTATION_SPEED * direction)
