from abc import ABC, abstractmethod
import math


class Vehicle(ABC):
    """Base class for all vehicles (player and AI)."""

    MAX_SPEED = 10.0
    ACCELERATION = 0.5
    FRICTION = 0.95
    ROTATION_SPEED = 3.0
    
    def __init__(self, x, y, width=30, height=15):
        self.__x = float(x)
        self.__y = float(y)
        self.__speed = 0.0
        self.__angle = 0.0
        self._width = width
        self._height = height
        self._color = (255, 255, 255)
    
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    @property
    def speed(self):
        return self.__speed
    
    @property
    def angle(self):
        return self.__angle
    
    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height
    
    @property
    def color(self):
        return self._color
    
    def set_position(self, x, y):
        self.__x = float(x)
        self.__y = float(y)
    
    def set_angle(self, angle):
        self.__angle = float(angle)

    def accelerate(self, amount):
        self.__speed += amount
        if self.__speed > self.MAX_SPEED:
            self.__speed = self.MAX_SPEED
        elif self.__speed < -self.MAX_SPEED * 0.5:
            self.__speed = -self.MAX_SPEED * 0.5
    
    def rotate(self, amount):
        self.__angle += amount
    
    def apply_friction(self):
        self.__speed *= self.FRICTION
        if abs(self.__speed) < 0.1:
            self.__speed = 0.0
    
    def update_position(self, dt):
        if abs(self.__speed) > 0.01:
            rad = math.radians(self.__angle)
            self.__x += math.cos(rad) * self.__speed
            self.__y += math.sin(rad) * self.__speed
    
    def get_corners(self):
        # Standard rectangle shape
        rad = math.radians(self.__angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        hw = self._width / 2
        hh = self._height / 2
        
        # Rectangle corners
        corners = [
            (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
        ]
        
        rotated = []
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a
            ry = cx * sin_a + cy * cos_a
            rotated.append((self.__x + rx, self.__y + ry))
        
        return rotated

    def get_raycast_angles(self):
        """Return raycast angles relative to vehicle direction (7 rays)."""
        return [-90, -60, -30, 0, 30, 60, 90]

    def get_raycasts(self, track, max_distance=300):
        """
        Perform raycasts and return distances to walls.
        Also returns endpoints for visualization.
        """
        ray_angles = self.get_raycast_angles()
        distances = []
        endpoints = []

        for relative_angle in ray_angles:
            absolute_angle = self.__angle + relative_angle
            distance = track.cast_ray(self.__x, self.__y, absolute_angle, max_distance)
            distances.append(distance)

            rad = math.radians(absolute_angle)
            end_x = self.__x + math.cos(rad) * distance
            end_y = self.__y + math.sin(rad) * distance
            endpoints.append((end_x, end_y))

        return distances, endpoints

    @abstractmethod
    def handle_input(self):
        pass
    
    def update(self, dt):
        self.handle_input()
        self.apply_friction()
        self.update_position(dt)
