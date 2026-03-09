import pygame
from entities.player_car import PlayerCar
from core.track import Track
from core.track_loader import TrackLoader
from core.renderer import Renderer
from core.physics_engine import PhysicsEngine
from core.lap_timer import LapTimer


class GameEngine:
    """Main game loop and logic controller."""

    def __init__(self, width=1200, height=800, track_file=None):
        """
        Initialize game engine.

        Args:
            width: Window width
            height: Window height
            track_file: Path to PNG track file (optional, uses default track if None)
        """
        pygame.init()
        self._width = width
        self._height = height
        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Racing Game")
        
        self._clock = pygame.time.Clock()
        self._running = True
        self._fps = 60
        self._fixed_dt = 1 / 60  # Fixed timestep matching AI environment

        # Load track
        if track_file:
            loader = TrackLoader()
            track_data = loader.load_from_png(track_file)
            self._track = Track(track_data=track_data)
            # Update window size to match track
            self._width = track_data['width']
            self._height = track_data['height']
            self._screen = pygame.display.set_mode((self._width, self._height))
        else:
            self._track = Track(width, height)

        self._renderer = Renderer(self._screen)
        self._physics = PhysicsEngine()
        self._lap_timer = LapTimer()

        start_x, start_y = self._track.start_position
        self._player = PlayerCar(start_x, start_y)
        self._player.set_angle(90)
        self._vehicles = [self._player]

        # Checkpoint tracking
        self._next_checkpoint = 0
        self._collision_count = 0
        self._show_checkpoints = False  # Toggle with C key
        self._show_raycasts = False  # Toggle with V key
        self._cached_raycast_endpoints = None
        self._max_raycast_distance = 500  # Same as AI environment

        # Start race
        self._lap_timer.start_race()

    def run(self):
        """Main game loop."""
        while self._running:
            self._clock.tick(self._fps)

            self._handle_events()
            self._update(self._fixed_dt)
            self._render()
        
        pygame.quit()
    
    def _reset_race(self):
        """Reset race to initial state."""
        start_x, start_y = self._track.start_position
        self._player.set_position(start_x, start_y)
        self._player.set_angle(90)
        self._player.accelerate(-self._player.speed)
        self._next_checkpoint = 0
        self._collision_count = 0
        self._track.reset_checkpoints()
        self._lap_timer.reset()
        self._lap_timer.start_race()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                elif event.key == pygame.K_r:
                    self._reset_race()
                elif event.key == pygame.K_c:
                    self._show_checkpoints = not self._show_checkpoints
                elif event.key == pygame.K_v:
                    self._show_raycasts = not self._show_raycasts

    def _update(self, dt):
        self._lap_timer.update()

        for vehicle in self._vehicles:
            old_x = vehicle.x
            old_y = vehicle.y
            
            vehicle.update(dt)

            # Handle collision FIRST (before checking checkpoints)
            # This prevents false checkpoint detections when position is reset
            collision_occurred = False
            if self._physics.handle_collision(vehicle, self._track):
                self._collision_count += 1
                if self._track.check_collision(vehicle.get_corners()):
                    vehicle.set_position(old_x, old_y)
                    collision_occurred = True

                # Only check checkpoints if no collision occurred (to avoid false detections)
            if not collision_occurred:
                if self._track.check_checkpoint_crossing(
                    old_x, old_y, vehicle.x, vehicle.y, self._next_checkpoint
                ):
                    self._next_checkpoint += 1

                # Check finish line only after all checkpoints are passed
                if self._next_checkpoint >= self._track.total_checkpoints:
                    if self._track.check_finish_line_crossing(old_x, old_y, vehicle.x, vehicle.y):
                        lap_info = self._lap_timer.complete_lap()
                        self._next_checkpoint = 0
                        self._track.reset_checkpoints()

        # Cache raycasts for rendering
        if self._show_raycasts:
            _, self._cached_raycast_endpoints = self._player.get_raycasts(
                self._track, self._max_raycast_distance
            )
        else:
            self._cached_raycast_endpoints = None

    def _render(self):
        self._renderer.clear(self._track.background_color)
        self._renderer.draw_track(self._track, self._show_checkpoints)

        for vehicle in self._vehicles:
            self._renderer.draw_vehicle(vehicle)

            # Draw raycasts if enabled
            if self._show_raycasts and self._cached_raycast_endpoints:
                self._renderer.draw_raycasts(vehicle, self._cached_raycast_endpoints)

        # Draw statistics
        y_offset = 10
        line_height = 35

        lap_text = f"Lap: {self._lap_timer.current_lap}"
        self._renderer.draw_text(lap_text, 10, y_offset)
        y_offset += line_height

        time_text = f"Time: {self._lap_timer.format_time(self._lap_timer.current_lap_time)}"
        self._renderer.draw_text(time_text, 10, y_offset)
        y_offset += line_height

        best_text = f"Best: {self._lap_timer.format_time(self._lap_timer.best_lap_time)}"
        self._renderer.draw_text(best_text, 10, y_offset)
        y_offset += line_height

        checkpoint_text = f"Checkpoint: {self._next_checkpoint}/{self._track.total_checkpoints}"
        self._renderer.draw_text(checkpoint_text, 10, y_offset)
        y_offset += line_height

        speed_text = f"Speed: {abs(self._player.speed):.1f}"
        self._renderer.draw_text(speed_text, 10, y_offset)
        y_offset += line_height

        collision_text = f"Collisions: {self._collision_count}"
        self._renderer.draw_text(collision_text, 10, y_offset)

        self._renderer.update_display()
