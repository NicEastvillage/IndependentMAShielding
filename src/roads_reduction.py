from typing import List, Tuple

from roads import RoadNetworkState, RoadNetwork, CarState, ChancedRoadNetworkState
from systems import LabelledTransitionSystem


class AbstractRoundaboutSystem(LabelledTransitionSystem[RoadNetwork]):
    def __init__(self, roundabout: RoadNetwork, agent: int):
        self._system = roundabout
        self._pov = agent

    def get_init_state(self) -> RoadNetworkState:
        s = self._system.get_init_state()
        return self._normalize(s, self._pov)

    def get_successor_states(self, state: RoadNetworkState, actions: Tuple[int, ...]) -> Tuple[ChancedRoadNetworkState, ...]:
        succs = self._system.get_successors(state, actions)
        return tuple(ChancedRoadNetworkState(succ.chance, self._normalize(succ.state, self._pov), self._system) for succ in succs)

    def get_abstract_successor_states(self, state: RoadNetworkState, action: int) -> List[RoadNetworkState]:
        assert len(state.cars) <= 2   # We can avoid recursion

        if len(state.cars) == 1:
            successors = self.get_successor_states(state, (action,))
            return [self._normalize(successor.state, self._pov) for successor in successors]

        other_id = 1 - self._pov
        action_vecs = [(action, a) for a in range(self._system.get_agent_action_count(state, other_id))] if self._pov == 0 else [
            (a, action) for a in range(self._system.get_agent_action_count(state, other_id))]

        successors = []
        for avec in action_vecs:
            chanced_successors = self.get_successor_states(state, avec)
            successors.extend([self._normalize(cs.state, self._pov) for cs in chanced_successors])

        return successors

    def get_agent_action_count(self, state: RoadNetworkState) -> int:
        return self._system.get_agent_action_count(state, self._pov)

    @staticmethod
    def _normalize(state: RoadNetworkState, agent: int) -> RoadNetworkState:
        assert len(state.system.roads) == 1
        if len(state.system.cars) == 1:
            return RoadNetworkState(state.system, (CarState(agent, 0, 0, state.cars[agent].vel),))

        ego_pos = state.cars[agent].pos
        ego_vel = state.cars[agent].vel

        car_count = len(state.cars)
        car_in_front = state.cars[(agent + 1) % car_count]
        dist = (car_in_front.pos - ego_pos) % state.system.roads[0].length
        for i, car in enumerate(state.cars):
            if i == agent:
                continue
            alt_dist = (car.pos - ego_pos) % state.system.roads[0].length
            if alt_dist <= dist:
                car_in_front = car
                dist = alt_dist

        abstract_ego_car = CarState(0, 0, 0, ego_vel)
        abstract_front_car = CarState(1, 0, dist, car_in_front.vel)

        return RoadNetworkState(state.system, (abstract_ego_car, abstract_front_car))


def roundabout_safety(min_dist: int = 2):
    def func(state: RoadNetworkState) -> bool:
        if len(state.cars) <= 1:
            return True
        dist = state.cars[1].pos - state.cars[0].pos
        return dist >= min_dist
    return func
