import numpy as np


class Track:

    def __init__(self, width=None, height=None, track_data=None):
        """
        Initialize track either from dimensions (creates default track)
        or from track_data dict (loaded from PNG).

        Args:
            width: Track width (for default track)
            height: Track height (for default track)
            track_data: Dict with walls, checkpoints, start_position, width, height (from TrackLoader)
        """
        if track_data:
            self._width = track_data['width']
            self._height = track_data['height']
            self._walls = track_data['walls']
            self._checkpoints = track_data['checkpoints']
            self._start_position = track_data['start_position']
            self._start_finish_line = track_data.get('start_finish_line', None)
        else:
            self._width = width or 1200
            self._height = height or 800
            self._walls = []
            self._checkpoints = []
            self._start_position = (100, height // 2 if height else 400)
            self._start_finish_line = None
            self._setup_basic_track()
            self._setup_checkpoints()

        self._background_color = (50, 50, 50)
        self._wall_color = (100, 100, 100)
        self._checkpoint_color = (255, 215, 0)

        # Build collision grid for fast raycasting
        self._collision_grid = np.zeros((self._height, self._width), dtype=np.bool_)
        for wall in self._walls:
            x0 = max(0, wall['x'])
            y0 = max(0, wall['y'])
            x1 = min(self._width, wall['x'] + wall['width'])
            y1 = min(self._height, wall['y'] + wall['height'])
            self._collision_grid[y0:y1, x0:x1] = True

    def _setup_basic_track(self):
        """Create basic rectangular track with outer walls and one inner obstacle."""
        wall_thickness = 20

        # Outer walls
        self._walls.append({
            'x': 0, 'y': 0,
            'width': self._width, 'height': wall_thickness
        })
        self._walls.append({
            'x': 0, 'y': self._height - wall_thickness,
            'width': self._width, 'height': wall_thickness
        })
        self._walls.append({
            'x': 0, 'y': 0,
            'width': wall_thickness, 'height': self._height
        })
        self._walls.append({
            'x': self._width - wall_thickness, 'y': 0,
            'width': wall_thickness, 'height': self._height
        })

        # Inner obstacle
        self._walls.append({
            'x': self._width // 2 - 100, 'y': self._height // 2 - 50,
            'width': 200, 'height': 100
        })

    def _setup_checkpoints(self):
        """Set up default checkpoints for basic track."""
        margin = 50
        # Right side checkpoint
        self._checkpoints.append({
            'x1': self._width - 200, 'y1': margin,
            'x2': self._width - 200, 'y2': self._height - margin,
            'id': 0,
            'passed': False
        })

        # Top checkpoint
        self._checkpoints.append({
            'x1': self._width - margin, 'y1': 150,
            'x2': margin, 'y2': 150,
            'id': 1,
            'passed': False
        })

        # Left side checkpoint
        self._checkpoints.append({
            'x1': 200, 'y1': margin,
            'x2': 200, 'y2': self._height - margin,
            'id': 2,
            'passed': False
        })

        # Bottom checkpoint (finish line)
        self._checkpoints.append({
            'x1': self._width - margin, 'y1': self._height - 150,
            'x2': margin, 'y2': self._height - 150,
            'id': 3,
            'passed': False
        })

    def check_checkpoint_crossing(self, prev_x, prev_y, curr_x, curr_y, next_checkpoint_id):
        """Check if vehicle crossed the next checkpoint."""
        if next_checkpoint_id >= len(self._checkpoints):
            return False

        checkpoint = self._checkpoints[next_checkpoint_id]

        # Skip if already passed
        if checkpoint.get('passed', False):
            return False

        cp_x1, cp_y1 = checkpoint['x1'], checkpoint['y1']
        cp_x2, cp_y2 = checkpoint['x2'], checkpoint['y2']

        # Simple line-line intersection using determinant method
        # Line 1: from (prev_x, prev_y) to (curr_x, curr_y)
        # Line 2: from (cp_x1, cp_y1) to (cp_x2, cp_y2)

        def line_intersection(p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
            """
            Check if line segment p0-p1 intersects with line segment p2-p3
            Returns True if they intersect
            """
            s1_x = p1_x - p0_x
            s1_y = p1_y - p0_y
            s2_x = p3_x - p2_x
            s2_y = p3_y - p2_y

            denom = (-s2_x * s1_y + s1_x * s2_y)
            if abs(denom) < 1e-10:  # Lines are parallel
                return False

            s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / denom
            t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / denom

            if 0 <= s <= 1 and 0 <= t <= 1:
                return True
            return False

        # Check if movement line crosses checkpoint line
        intersects = line_intersection(prev_x, prev_y, curr_x, curr_y, cp_x1, cp_y1, cp_x2, cp_y2)

        if intersects:
            checkpoint['passed'] = True
            return True

        return False

    def check_finish_line_crossing(self, prev_x, prev_y, curr_x, curr_y):
        """Check if vehicle crossed the finish line (any direction)."""
        if not self._start_finish_line:
            return False

        # Line intersection algorithm
        def ccw(ax, ay, bx, by, cx, cy):
            return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

        def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
            """Check if line (x1,y1)-(x2,y2) intersects with line (x3,y3)-(x4,y4)"""
            a = ccw(x1, y1, x3, y3, x4, y4)
            b = ccw(x2, y2, x3, y3, x4, y4)
            c = ccw(x1, y1, x2, y2, x3, y3)
            d = ccw(x1, y1, x2, y2, x4, y4)
            return (a != b) and (c != d)

        fl_x1, fl_y1 = self._start_finish_line['x1'], self._start_finish_line['y1']
        fl_x2, fl_y2 = self._start_finish_line['x2'], self._start_finish_line['y2']

        # Check if movement line crosses finish line (any direction)
        return lines_intersect(prev_x, prev_y, curr_x, curr_y, fl_x1, fl_y1, fl_x2, fl_y2)

    def reset_checkpoints(self):
        """Reset all checkpoints to not passed."""
        for checkpoint in self._checkpoints:
            checkpoint['passed'] = False

    @property
    def walls(self):
        return self._walls

    @property
    def checkpoints(self):
        return self._checkpoints

    @property
    def start_position(self):
        return self._start_position

    @property
    def start_finish_line(self):
        return self._start_finish_line

    @property
    def total_checkpoints(self):
        return len(self._checkpoints)

    @property
    def background_color(self):
        return self._background_color

    @property
    def wall_color(self):
        return self._wall_color

    @property
    def checkpoint_color(self):
        return self._checkpoint_color

    def check_collision(self, corners):
        """Check if any corner collides with walls using collision grid."""
        w = self._width
        h = self._height
        grid = self._collision_grid
        for corner_x, corner_y in corners:
            ix = int(corner_x)
            iy = int(corner_y)
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                return True
            if grid[iy, ix]:
                return True
        return False

    def cast_ray(self, start_x, start_y, angle_deg, max_distance=300):
        """
        Cast a ray from start position at given angle.
        Returns distance to nearest wall (or max_distance if no hit).
        Uses collision grid for fast lookup.
        """
        import math
        rad = math.radians(angle_deg)
        dx = math.cos(rad)
        dy = math.sin(rad)

        step = 3
        distance = 0
        w = self._width
        h = self._height
        grid = self._collision_grid

        while distance < max_distance:
            check_x = int(start_x + dx * distance)
            check_y = int(start_y + dy * distance)

            if check_x < 0 or check_x >= w or check_y < 0 or check_y >= h:
                return distance

            if grid[check_y, check_x]:
                return distance

            distance += step

        return max_distance

