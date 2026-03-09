from entities.vehicle import Vehicle


class AICar(Vehicle):
    """AI-controlled vehicle."""

    def __init__(self, x, y):
        super().__init__(x, y)
        self._color = (255, 100, 0)  # Orange for AI
        self._current_action = 0

    def set_action(self, action):
        """
        Set action to execute.

        Actions (0-4):
            0: Nothing
            1: Gas
            2: Gas + Left
            3: Gas + Right
            4: Reverse
        """
        self._current_action = int(action)

    def handle_input(self):
        """Execute action set by AI."""
        action = self._current_action

        if action == 1:  # Gas
            self.accelerate(self.ACCELERATION)
        elif action == 2:  # Gas + Left
            self.accelerate(self.ACCELERATION)
            if abs(self.speed) > 0.5:
                self.rotate(-self.ROTATION_SPEED)
        elif action == 3:  # Gas + Right
            self.accelerate(self.ACCELERATION)
            if abs(self.speed) > 0.5:
                self.rotate(self.ROTATION_SPEED)
        elif action == 4:  # Reverse
            self.accelerate(-self.ACCELERATION)
