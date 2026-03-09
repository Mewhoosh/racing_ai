import math


class PhysicsEngine:
    
    def __init__(self):
        self._collision_response = 0.8

    def handle_collision(self, vehicle, track):
        """Check collision and push vehicle away from walls."""
        corners = vehicle.get_corners()
        
        if track.check_collision(corners):
            # Find which corners collided and calculate push vector
            push_x, push_y = self._calculate_push_vector(corners, track)

            if push_x == 0 and push_y == 0:
                # Fallback - push backwards
                rad = math.radians(vehicle.angle)
                push_x = -math.cos(rad) * 10
                push_y = -math.sin(rad) * 10

            new_x = vehicle.x + push_x
            new_y = vehicle.y + push_y

            vehicle.set_position(new_x, new_y)
            vehicle.accelerate(-vehicle.speed)  # Full stop on collision

            return True
        return False

    def _calculate_push_vector(self, corners, track):
        """Calculate push vector based on corner collisions with walls."""
        total_push_x = 0.0
        total_push_y = 0.0
        collision_count = 0

        for corner_x, corner_y in corners:
            for wall in track.walls:
                wx, wy = wall['x'], wall['y']
                ww, wh = wall['width'], wall['height']

                if wx <= corner_x <= wx + ww and wy <= corner_y <= wy + wh:
                    # Corner inside wall - calculate exit direction
                    # Find nearest wall edge
                    dist_left = corner_x - wx
                    dist_right = (wx + ww) - corner_x
                    dist_top = corner_y - wy
                    dist_bottom = (wy + wh) - corner_y

                    min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
                    push_amount = min_dist + 2  # Margin to exit wall

                    if min_dist == dist_left:
                        total_push_x -= push_amount
                    elif min_dist == dist_right:
                        total_push_x += push_amount
                    elif min_dist == dist_top:
                        total_push_y -= push_amount
                    else:
                        total_push_y += push_amount

                    collision_count += 1

        if collision_count > 0:
            return total_push_x / collision_count, total_push_y / collision_count
        return 0, 0

