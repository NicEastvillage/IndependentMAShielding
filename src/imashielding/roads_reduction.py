import heapq
from itertools import product
from typing import Tuple

from imashielding.roads import RoadNetworkState, RoadNetwork, CarState
from imashielding.systems import LabelledTransitionSystem, ChancedState, StateSafety


AbstractRoundaboutState = RoadNetworkState

class AbstractRoundaboutSystem(LabelledTransitionSystem[AbstractRoundaboutState]):
    def __init__(self, roundabout: RoadNetwork):
        self._system = roundabout

    def get_init_state(self) -> AbstractRoundaboutState:
        s = self._system.get_init_state()
        return self.normalize(s, 0)

    def get_abstract_successor_states(self, state: AbstractRoundaboutState, action: int) -> Tuple[AbstractRoundaboutState, ...]:
        assert len(state.cars) <= 2

        if len(state.cars) == 1:
            chanced_successors = self._system.get_successors(state, (action,))
            return tuple(self.normalize(cs.state, 0) for cs in chanced_successors)

        action_vecs = tuple((action, a) for a in range(self._system.get_agent_action_count(state, 1)))

        successors = []
        for av in action_vecs:
            chanced_successors = self._system.get_successors(state, av)
            successors.extend([self.normalize(cs.state, 0) for cs in chanced_successors])

        return tuple(successors)

    def get_action_count(self, state: AbstractRoundaboutState) -> int:
        # Assumption: All cars have the same actions available
        return self._system.get_agent_action_count(state, 0)

    def normalize(self, state: RoadNetworkState, agent: int) -> AbstractRoundaboutState:
        assert len(self._system.roads) == 1
        if len(self._system.cars) == 1:
            return AbstractRoundaboutState((CarState(agent, 0, 0, state.cars[agent].vel, False, False),))

        ego_pos = state.cars[agent].pos
        ego_vel = state.cars[agent].vel

        car_count = len(state.cars)
        car_in_front = state.cars[(agent + 1) % car_count]
        dist = (car_in_front.pos - ego_pos) % self._system.roads[0].length
        for i, car in enumerate(state.cars):
            if i == agent:
                continue
            alt_dist = (car.pos - ego_pos) % self._system.roads[0].length
            if alt_dist <= dist:
                car_in_front = car
                dist = alt_dist

        abstract_ego_car = CarState(0, 0, 0, ego_vel, False, False)
        abstract_front_car = CarState(1, 0, dist, car_in_front.vel, False, False)

        return AbstractRoundaboutState((abstract_ego_car, abstract_front_car))


def roundabout_safety(min_dist: int = 4):
    def func(state: RoadNetworkState) -> StateSafety:
        if len(state.cars) <= 1:
            return StateSafety.SAFE
        dist = state.cars[1].pos - state.cars[0].pos
        return StateSafety.SAFE if dist >= min_dist else StateSafety.UNSAFE
    return func



AbstractRoadNetworkState = RoadNetworkState

class AbstractRoadNetworkSystem(LabelledTransitionSystem[AbstractRoadNetworkState]):
    def __init__(self, roundabout: RoadNetwork):
        self._system = roundabout

    def get_init_state(self) -> AbstractRoadNetworkState:
        # TODO: There could be more than one initial state based on POV
        s = self._system.get_init_state()
        return self.normalize(s, 0)

    def get_abstract_successor_states(self, state: AbstractRoadNetworkState, action: int) -> Tuple[AbstractRoadNetworkState, ...]:
        car_count = len(state.cars)
        if car_count == 1:
            chanced_successors = self._system.get_successors(state, (action,))
            return tuple(self.normalize(cs.state, 0) for cs in chanced_successors)

        other_ranges = (range(self._system.get_agent_action_count(state, i)) for i in range(1, car_count))
        action_vecs = ((action,) + combo for combo in product(*other_ranges))

        successors = []
        for av in action_vecs:
            chanced_successors = self._system.get_successors(state, av)
            successors.extend([self.normalize(cs.state, 0) for cs in chanced_successors])
        return tuple(successors)

    def get_action_count(self, state: AbstractRoadNetworkState) -> int:
        # Assumption: All cars have the same actions available
        return self._system.get_agent_action_count(state, 0)

    def normalize(self, state: RoadNetworkState, agent: int) -> AbstractRoadNetworkState:
        if len(state.cars) == 1:
            car = state.cars[agent]
            return AbstractRoadNetworkState((CarState(0, car.road, car.pos, car.vel, False, False),))

        ego = state.cars[agent]
        roads = self._system.roads

        # Dijkstra over segments: dist_start[s] = distance (in lengths) from the ego
        # car's position to reach position 0 of segment s, traversing segment ends.
        dist_start = [float("inf")] * len(roads)
        pq = []

        def relax(seg: int, dist: float) -> None:
            if dist < dist_start[seg]:
                dist_start[seg] = dist
                heapq.heappush(pq, (dist, seg))

        # From the ego's position, travel the rest of its segment (length - pos) and
        # arrive at position 0 of the following segment(s). Splits enqueue both ends;
        # merges (end_left == end_right) enqueue that end once (via the set).
        ego_segment = roads[ego.road]
        for end in {ego_segment.end_left, ego_segment.end_right}:
            relax(end, ego_segment.length - ego.pos)

        while pq:
            dist, seg = heapq.heappop(pq)
            if dist > dist_start[seg]:
                continue
            segment = roads[seg]
            for end in {segment.end_left, segment.end_right}:
                relax(end, dist + segment.length)

        def distance_ahead(car: CarState) -> float:
            # Ahead on the ego's own segment: the simple forward gap.
            if car.road == ego.road and car.pos >= ego.pos:
                return car.pos - ego.pos
            # Otherwise loop around: reach the start of the car's segment, then travel
            # to its position. Handles cars behind the ego on the same segment too.
            return dist_start[car.road] + car.pos

        cars_by_dist = sorted(
            ((distance_ahead(car), car) for car in state.cars),
            key=lambda t: (t[0], t[1].id),
        )

        new_cars = ()
        use_symmetry = roads[ego.road].symmetric_to is not None and roads[ego.road].symmetric_to < ego.road
        if use_symmetry:
            for i, (_, car) in enumerate(cars_by_dist):
                new_cars += (CarState(i, roads[car.road].symmetric_to, car.pos, car.vel, car.just_merged, False),)
        else:
            for i, (_, car) in enumerate(cars_by_dist):
                new_cars += (CarState(i, car.road, car.pos, car.vel, car.just_merged, False),)

        return AbstractRoadNetworkState(new_cars)


def road_safety(min_dist: int = 3, merge_space: int = 8):
    def func(state: RoadNetworkState) -> StateSafety:
        if len(state.cars) <= 1:
            return StateSafety.SAFE

        # Property: distToNonMergingCarInFront >= min_dist && just_merged implies distToCarBehind >= merge_space

        somebody_else_messed_up = False

        # Assumption: Cars are sorted by dist to ego (id 0)
        for i in range(len(state.cars)):    # FIXME: IT IS LIKELY NOT ENOUGH TO ONLY CHECK CONSECUTIVE PAIRS :(
            behind_car = state.cars[i]
            front_car = state.cars[(i + 1) % len(state.cars)]

            if behind_car.road == front_car.road and behind_car.pos == front_car.pos:
                # Merging car or car with highest vel messed up
                if front_car.just_merged:
                    at_fault = front_car
                elif behind_car.just_merged:
                    at_fault = behind_car
                else:
                    at_fault = behind_car if behind_car.vel > front_car.vel else front_car
                if at_fault.id == 0:
                    return StateSafety.UNSAFE
                somebody_else_messed_up = True

            if front_car.just_merged:
                # Front car must ensure space behind him on priority road
                if behind_car.road == front_car.road:
                    dist = front_car.pos - behind_car.pos
                    if 0 <= dist < merge_space:
                        if front_car.id == 0:
                            return StateSafety.UNSAFE   # Ego messed up
                        somebody_else_messed_up = True
                else:
                    # Assumption: Road segments are longer than merge_space
                    behind_road = state.system.roads[behind_car.road]
                    front_road = state.system.roads[front_car.road]
                    if front_car.pos <= merge_space:
                        continue
                    if front_road.start != behind_road.id:
                        continue
                    dist = behind_road.length - behind_car.pos + front_car.pos
                    if 0 <= dist < merge_space:
                        if front_car.id == 0:
                            return StateSafety.UNSAFE   # Ego messed up
                        somebody_else_messed_up = True

            else:
                # Behind car must ensure space ahead of him
                if behind_car.road == front_car.road:
                    dist = front_car.pos - behind_car.pos
                    if 0 <= dist < min_dist:
                        if behind_car.id == 0:
                            return StateSafety.UNSAFE   # Ego messed up
                        somebody_else_messed_up = True
                else:
                    # Assumption: Road segments are longer than min_dist
                    behind_road = state.system.roads[behind_car.road]
                    seg_remaining = behind_road.length - behind_car.pos
                    if seg_remaining <= min_dist:
                        continue
                    if front_car.road != behind_road.end_left and front_car.road != behind_road.end_right:
                        continue
                    dist = seg_remaining + front_car.pos
                    if 0 <= dist < min_dist:
                        if behind_car.id == 0:
                            return StateSafety.UNSAFE   # Ego messed up
                        somebody_else_messed_up = True

        return StateSafety.SAFE_RELEASE if somebody_else_messed_up else StateSafety.SAFE

    return func
