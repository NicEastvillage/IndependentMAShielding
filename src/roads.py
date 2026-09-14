import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from systems import ChancedState, ConcurrentGame


@dataclass(frozen=True, slots=True)
class CarState:
    id: int
    road: int
    pos: int
    vel: int
    just_merged: bool
    braking: bool


@dataclass(frozen=True, slots=True)
class RoadNetworkState:
    system: "RoadNetwork"
    cars: Tuple[CarState, ...]

    def get_successors(self, actions: Tuple[int, ...]) -> Tuple[ChancedState["RoadNetworkState"], ...]:
        return self.system.get_successors(self, actions)


@dataclass(frozen=True, slots=True)
class CarDef:
    id: int
    init_road: int
    init_pos: int
    init_vel: int = 0


@dataclass(frozen=True, slots=True)
class Arc:
    """Visual placement of a road segment: circular arc traversed counter-clockwise."""
    center: Tuple[float, float]
    radius: float
    start: float   # radians
    sweep: float   # radians

    def frame(self, t: float) -> Tuple[Tuple[float, float], float]:
        """World position and travel heading at fraction t in [0, 1) along the arc."""
        angle = self.start + t * self.sweep
        return ((self.center[0] + self.radius * math.cos(angle),
                 self.center[1] + self.radius * math.sin(angle)),
                angle + math.pi / 2)


@dataclass(frozen=True, slots=True)
class RoadSegment:
    id: int
    length: int
    end_left: int
    end_right: int
    start: int
    symmetric_to: Optional[int] = None
    arc: Optional[Arc] = None


@dataclass(frozen=True, slots=True)
class ChancedOutcomeAccel:
    chance: float
    accel: int


class RoadNetwork(ConcurrentGame[RoadNetworkState]):
    def __init__(self, roads: Tuple[RoadSegment, ...], cars: Tuple[CarDef, ...], max_vel: int, min_vel: int, accel_actions: Tuple[Tuple[ChancedOutcomeAccel, ...], ...]):
        self.roads: Tuple[RoadSegment, ...] = roads
        self.cars: Tuple[CarDef, ...] = cars
        self.max_vel = max_vel
        self.min_vel = min_vel
        self.accel_actions = accel_actions   # Indexes are actions; Entries are tuples of weighted outcome accelerations

        # Validation
        assert min_vel <= max_vel

        assert len(roads) > 0
        for i, road in enumerate(roads):
            assert road.id == i
            assert 0 < road.length
            assert 0 <= road.start < len(roads)
            assert 0 <= road.end_left < len(roads)
            assert 0 <= road.end_right < len(roads)
            assert road.arc is None or road.arc.radius > 0

        assert len(cars) > 0
        for i, car in enumerate(cars):
            assert car.id == i
            assert 0 <= car.init_road < len(roads)
            assert 0 <= car.init_pos < roads[car.init_road].length
            assert min_vel <= car.init_vel <= max_vel

    @staticmethod
    def create_roundabout_scenario(car_count: int = 6, road_length: int = 32, max_vel: int = 4) -> "RoadNetwork":
        road = RoadSegment(0, road_length, 0, 0, 0, Arc((0.0, 0.0), 1.0, -math.pi / 2, 2 * math.pi))
        init_dist = road_length / car_count
        cars = tuple(CarDef(i, 0, int(init_dist * i)) for i in range(car_count))
        # Acceleration is usually within [-1..1] but there is a 20% chance for a 100% increase
        accel_actions = (
            (
                ChancedOutcomeAccel(0.2, -2),
                ChancedOutcomeAccel(0.8, -1),
            ),
            (
                ChancedOutcomeAccel(1.0, 0),
            ),
            (
                ChancedOutcomeAccel(0.8, 1),
                ChancedOutcomeAccel(0.2, 2),
            )
        )
        return RoadNetwork((road,), cars, max_vel, 0, accel_actions)

    @staticmethod
    def create_double_roundabout_scenario(car_count: int = 6, road_per_roundabout: int = 32, max_vel: int = 4) -> "RoadNetwork":
        road_dist = 2.4
        straight_radius = 10.0
        straight_y_offset = math.sqrt(straight_radius**2 - (road_dist / 2)**2)
        straight_sweep = math.acos(straight_y_offset / straight_radius)
        roads = (
            RoadSegment(0, round(road_per_roundabout / 2), 1, 1, 1, symmetric_to=3, arc=Arc((0.0, 0.0), 1.0, -math.pi / 2, math.pi)),
            RoadSegment(1, round(road_per_roundabout / 2), 0, 5, 0, symmetric_to=2, arc=Arc((0.0, 0.0), 1.0, math.pi / 2, math.pi)),
            RoadSegment(2, round(road_per_roundabout / 2), 3, 4, 3, symmetric_to=1, arc=Arc((road_dist, 0.0), 1.0, -math.pi / 2, math.pi)),
            RoadSegment(3, round(road_per_roundabout / 2), 2, 2, 2, symmetric_to=0, arc=Arc((road_dist, 0.0), 1.0, math.pi / 2, math.pi)),
            RoadSegment(4, round(road_dist * road_per_roundabout / math.pi), 1, 1, 2, symmetric_to=5, arc=Arc((road_dist / 2, -straight_y_offset + 1.0), straight_radius, math.pi / 2 - straight_sweep, 2 * straight_sweep)),
            RoadSegment(5, round(road_dist * road_per_roundabout / math.pi), 2, 2, 1, symmetric_to=4, arc=Arc((road_dist / 2, straight_y_offset - 1.0), straight_radius, -math.pi / 2 - straight_sweep, 2 * straight_sweep)),
        )
        cars = tuple(CarDef(i, i % len(roads), 2 + int(road_per_roundabout * i / len(roads) / 4)) for i in range(car_count))
        # Acceleration is usually within [-1..1] but there is a 20% chance for a 100% increase
        accel_actions = (
            (
                ChancedOutcomeAccel(0.2, -2),
                ChancedOutcomeAccel(0.8, -1),
            ),
            (
                ChancedOutcomeAccel(1.0, 0),
            ),
            (
                ChancedOutcomeAccel(0.8, 1),
                ChancedOutcomeAccel(0.2, 2),
            )
        )
        return RoadNetwork(roads, cars, max_vel, 0, accel_actions)

    def get_init_state(self) -> RoadNetworkState:
        cars = tuple(CarState(c.id, c.init_road, c.init_pos, c.init_vel, False, False) for c in self.cars)
        return RoadNetworkState(self, cars)

    def get_successors(self, state: RoadNetworkState, actions: Tuple[int, ...]) -> Tuple[ChancedState[RoadNetworkState], ...]:
        assert len(state.cars) == len(actions)

        # Partial joint successors; cars [0..i) have already been moved
        partials: List[Tuple[float, Tuple[CarState, ...]]] = [(1.0, tuple(state.cars))]
        for i, action in enumerate(actions):
            expanded: Dict[Tuple[CarState, ...], float] = {}
            for chance, cars in partials:
                car = cars[i]
                for outcome in self.accel_actions[action]:
                    vel = min(max(car.vel + outcome.accel, self.min_vel), self.max_vel)
                    for move_chance, road, pos, just_merged in self._advance(car.road, car.pos, vel):
                        new_cars = cars[:i] + (CarState(i, road, pos, vel, just_merged, outcome.accel < 0),) + cars[i + 1:]
                        expanded[new_cars] = expanded.get(new_cars, 0.0) + chance * outcome.chance * move_chance
            partials = [(chance, cars) for cars, chance in expanded.items()]
        return tuple(ChancedState[RoadNetworkState](chance, RoadNetworkState(self, tuple(cars))) for chance, cars in partials)

    def _advance(self, road: int, pos: int, vel: int) -> Iterator[Tuple[float, int, int, bool]]:
        """Yields (chance, road, pos, just_merged) placements after moving `vel` cells start from (road, pos).

        Completing a segment moves the car onto the right of left exit at random.
        """
        segment = self.roads[road]
        new_pos = pos + vel
        if new_pos < segment.length:
            yield 1.0, road, new_pos, False
            return
        remaining = new_pos - segment.length
        next_segs = ((segment.end_left, 1.0),) if segment.end_left == segment.end_right else ((segment.end_left, 0.5), (segment.end_right, 0.5))
        for next_seg, chance in next_segs:
            just_merged = segment.end_left == segment.end_right and segment.id != self.roads[next_seg].start
            for sub_chance, sub_road, sub_pos, jm in self._advance(next_seg, 0, remaining):
                yield chance * sub_chance, sub_road, sub_pos, just_merged

    def get_agent_action_count(self, state: RoadNetworkState, agent: int) -> int:
        return len(self.accel_actions)
