import math

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, slots=True)
class Arc:
    """Visual placement of an arced road segment, traversed counter-clockwise unless sweep is negative."""
    center: Tuple[float, float]
    radius: float
    start: float   # radians
    sweep: float   # radians

    def frame(self, t: float) -> Tuple[Tuple[float, float], float]:
        """World position and travel heading at fraction t in [0, 1) along the arc."""
        angle = self.start + t * self.sweep
        facing = angle + math.pi / 2 if self.sweep >= 0 else angle - math.pi / 2
        return ((self.center[0] + self.radius * math.cos(angle),
                 self.center[1] + self.radius * math.sin(angle)),
                facing)

@dataclass(frozen=True, slots=True)
class Straight:
    """Visual placement of a straight road segment."""
    start: Tuple[float, float]
    end: Tuple[float, float]

    def frame(self, t: float) -> Tuple[Tuple[float, float], float]:
        """World position and travel heading at fraction t in [0, 1) along the straight."""
        angle = math.atan2(self.end[1] - self.start[1], self.end[0] - self.start[0])
        return ((self.start[0] + t * (self.end[0] - self.start[0]),
                 self.start[1] + t * (self.end[1] - self.start[1])),
                angle)


Shape = Arc | Straight