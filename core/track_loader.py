from PIL import Image
import numpy as np
import json
import os


class TrackLoader:
    """
    Loads racing tracks from PNG images.

    Track format:
    - Black (0,0,0): Walls
    - White (255,255,255): Road
    - Yellow (255,255,0): Start/Finish line
    - Green (0,255,0): Checkpoint 0 (first)
    - Blue (0,0,255): Checkpoint 1 (second)
    - Red (255,0,0): Checkpoint 2 (third)
    """

    def __init__(self):
        self.wall_color = (0, 0, 0)
        self.road_color = (255, 255, 255)
        # Checkpoint colors in order: Green -> Blue -> Red
        self.checkpoint_colors = [
            (0, 255, 0),    # Green - Checkpoint 0
            (0, 0, 255),    # Blue - Checkpoint 1
            (255, 0, 0),    # Red - Checkpoint 2
        ]
        self.finish_line_color = (255, 255, 0)  # Yellow

    def load_from_png(self, filepath):
        """
        Load track from PNG. First checks if cached JSON exists.
        If not, processes PNG and creates cache.
        """
        cache_file = filepath.replace('.png', '_cache.json')

        # Try to load from cache first
        if os.path.exists(cache_file):
            png_time = os.path.getmtime(filepath)
            cache_time = os.path.getmtime(cache_file)

            # Use cache if it's newer than PNG
            if cache_time >= png_time:
                with open(cache_file, 'r') as f:
                    return json.load(f)

        # No cache or outdated - process PNG
        print(f"Processing {filepath}...")
        track_data = self._process_png(filepath)

        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(track_data, f)
        print(f"Cached to {cache_file}")

        return track_data

    def _process_png(self, filepath):
        """Process PNG file completely - all pixels."""
        img = Image.open(filepath).convert('RGB')
        width, height = img.size
        pixels = np.array(img)

        walls = []
        checkpoints = []
        start_position = (100, height // 2)
        start_finish_line = None

        # Find yellow start/finish line
        yellow_pixels = []
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[y, x]
                if abs(r - 255) < 30 and abs(g - 255) < 30 and b < 50:
                    yellow_pixels.append((x, y))

        if yellow_pixels:
            avg_x = sum(p[0] for p in yellow_pixels) // len(yellow_pixels)
            avg_y = sum(p[1] for p in yellow_pixels) // len(yellow_pixels)
            start_position = (avg_x, avg_y)
            x1, y1 = yellow_pixels[0]
            x2, y2 = yellow_pixels[-1]
            start_finish_line = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

        # Extract walls - group adjacent black pixels
        visited = np.zeros((height, width), dtype=bool)

        for y in range(height):
            for x in range(width):
                if visited[y, x]:
                    continue

                r, g, b = pixels[y, x]

                if r < 50 and g < 50 and b < 50:
                    # Found wall pixel, create rectangle
                    wall_rect = self._extract_wall_rect(pixels, visited, x, y, width, height)
                    if wall_rect:
                        walls.append(wall_rect)

        # Extract checkpoints from green pixels
        checkpoints = self._extract_checkpoints(pixels, width, height, start_position)

        return {
            'walls': walls,
            'checkpoints': checkpoints,
            'start_position': start_position,
            'start_finish_line': start_finish_line,
            'width': width,
            'height': height
        }

    def _extract_wall_rect(self, pixels, visited, start_x, start_y, width, height):
        """Extract rectangular wall from starting position."""
        # Find max width
        rect_width = 0
        for x in range(start_x, width):
            r, g, b = pixels[start_y, x]
            if r < 50 and g < 50 and b < 50 and not visited[start_y, x]:
                rect_width += 1
            else:
                break

        if rect_width == 0:
            return None

        # Find max height
        rect_height = 0
        for y in range(start_y, height):
            valid = True
            for x in range(start_x, start_x + rect_width):
                if x >= width:
                    valid = False
                    break
                r, g, b = pixels[y, x]
                if not (r < 50 and g < 50 and b < 50) or visited[y, x]:
                    valid = False
                    break

            if valid:
                rect_height += 1
            else:
                break

        if rect_height == 0:
            return None

        # Mark as visited
        for y in range(start_y, start_y + rect_height):
            for x in range(start_x, start_x + rect_width):
                if y < height and x < width:
                    visited[y, x] = True

        return {
            'x': start_x,
            'y': start_y,
            'width': rect_width,
            'height': rect_height
        }

    def _extract_checkpoints(self, pixels, width, height, start_position):
        """Extract checkpoint lines from colored pixels (green, blue, red)."""
        # Dictionary to store checkpoints by color
        checkpoints_by_color = {}

        # Detect checkpoints for each color
        for checkpoint_id, color in enumerate(self.checkpoint_colors):
            visited = np.zeros((height, width), dtype=bool)
            checkpoint_lines = []

            target_r, target_g, target_b = color

            for y in range(height):
                for x in range(width):
                    if visited[y, x]:
                        continue

                    r, g, b = pixels[y, x]

                    # Check if pixel matches checkpoint color with specific logic for each color
                    is_match = self._is_color_match(r, g, b, target_r, target_g, target_b)

                    if is_match:
                        line = self._trace_checkpoint_line(pixels, visited, x, y, width, height, color)
                        if line and len(line) > 3:
                            # Find extreme points of the line
                            all_x = [p[0] for p in line]
                            all_y = [p[1] for p in line]
                            min_x, max_x = min(all_x), max(all_x)
                            min_y, max_y = min(all_y), max(all_y)

                            # Determine if line is more horizontal or vertical
                            if max_y - min_y > max_x - min_x:
                                # Vertical line
                                avg_x = sum(all_x) // len(all_x)
                                x1, y1 = avg_x, min_y
                                x2, y2 = avg_x, max_y
                            else:
                                # Horizontal line
                                avg_y = sum(all_y) // len(all_y)
                                x1, y1 = min_x, avg_y
                                x2, y2 = max_x, avg_y

                            checkpoint_lines.append({
                                'x1': x1, 'y1': y1,
                                'x2': x2, 'y2': y2,
                                'id': checkpoint_id,
                                'color': color,
                                'passed': False
                            })

            # Group nearby checkpoints for this color
            if checkpoint_lines:
                grouped = self._group_checkpoints(checkpoint_lines)
                if grouped:
                    # Take the first grouped checkpoint for this color
                    checkpoints_by_color[checkpoint_id] = grouped[0]
                    checkpoints_by_color[checkpoint_id]['id'] = checkpoint_id

        # Convert to sorted list by ID
        checkpoints = []
        for i in range(len(self.checkpoint_colors)):
            if i in checkpoints_by_color:
                checkpoints.append(checkpoints_by_color[i])

        return checkpoints

    def _group_checkpoints(self, checkpoints):
        """Group nearby checkpoints that are part of the same line."""
        if not checkpoints:
            return []

        # Distance threshold for grouping (pixels)
        distance_threshold = 50

        grouped = []
        used = [False] * len(checkpoints)

        for i, cp1 in enumerate(checkpoints):
            if used[i]:
                continue

            # Start a new group
            group = [cp1]
            used[i] = True

            # Find all nearby checkpoints
            for j, cp2 in enumerate(checkpoints):
                if used[j]:
                    continue

                # Check if cp2 is close to any checkpoint in current group
                for cp in group:
                    # Calculate average distance between endpoints
                    dist1 = ((cp['x1'] - cp2['x1'])**2 + (cp['y1'] - cp2['y1'])**2)**0.5
                    dist2 = ((cp['x2'] - cp2['x2'])**2 + (cp['y2'] - cp2['y2'])**2)**0.5
                    avg_dist = (dist1 + dist2) / 2

                    if avg_dist < distance_threshold:
                        group.append(cp2)
                        used[j] = True
                        break

            # Merge group into single checkpoint
            if group:
                # Find extreme points
                all_x = [cp['x1'] for cp in group] + [cp['x2'] for cp in group]
                all_y = [cp['y1'] for cp in group] + [cp['y2'] for cp in group]

                # Calculate center line
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)

                # Determine if line is more horizontal or vertical
                width_span = max_x - min_x
                height_span = max_y - min_y

                if height_span > width_span:
                    # Vertical line - use average x
                    avg_x = sum(all_x) / len(all_x)
                    merged = {
                        'x1': int(avg_x), 'y1': min_y,
                        'x2': int(avg_x), 'y2': max_y,
                        'id': 0,
                        'passed': False
                    }
                else:
                    # Horizontal line - use average y
                    avg_y = sum(all_y) / len(all_y)
                    merged = {
                        'x1': min_x, 'y1': int(avg_y),
                        'x2': max_x, 'y2': int(avg_y),
                        'id': 0,
                        'passed': False
                    }

                grouped.append(merged)

        return grouped

    def _trace_checkpoint_line(self, pixels, visited, start_x, start_y, width, height, target_color):
        """Trace a line of colored pixels in both directions from start point."""
        target_r, target_g, target_b = target_color

        # Collect all connected pixels of the same color
        to_visit = [(start_x, start_y)]
        line = []
        local_visited = set()

        # Use flood fill to find all connected pixels
        while to_visit:
            x, y = to_visit.pop()

            if (x, y) in local_visited:
                continue
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            if visited[y, x]:
                continue

            r, g, b = pixels[y, x]
            # Use the same color matching logic
            if not self._is_color_match(r, g, b, target_r, target_g, target_b):
                continue

            local_visited.add((x, y))
            visited[y, x] = True
            line.append((x, y))

            # Add neighbors (prioritize straight directions)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                to_visit.append((x + dx, y + dy))

        return line

    def _is_color_match(self, r, g, b, target_r, target_g, target_b):
        """
        Check if pixel color matches target color.
        Uses specific logic for each checkpoint color to avoid conflicts.
        """
        tolerance = 60

        # Skip black pixels (walls)
        if r < 50 and g < 50 and b < 50:
            return False

        # Skip white pixels (road)
        if r > 200 and g > 200 and b > 200:
            return False

        # Skip yellow pixels (finish line)
        if r > 200 and g > 200 and b < 80:
            return False

        # Green checkpoint (0, 255, 0): G dominates, high G, low R and B
        if target_r == 0 and target_g == 255 and target_b == 0:
            return g > 150 and g > r + 50 and g > b + 50

        # Blue checkpoint (0, 0, 255): B dominates, high B, low R and G
        if target_r == 0 and target_g == 0 and target_b == 255:
            return b > 150 and b > r + 50 and b > g + 50

        # Red checkpoint (255, 0, 0): R dominates, high R, low G and B
        if target_r == 255 and target_g == 0 and target_b == 0:
            return r > 150 and r > g + 50 and r > b + 50

        # Fallback: general tolerance matching
        return (abs(r - target_r) < tolerance and
                abs(g - target_g) < tolerance and
                abs(b - target_b) < tolerance)




