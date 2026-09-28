from imashielding.roads import RoadNetwork
from imashielding.roads_reduction import AbstractRoadNetworkSystem, road_safety
from imashielding.shield import compute_shield
from imashielding.visualize import visualize


if __name__ == '__main__':
    net = RoadNetwork.create_double_roundabout_scenario(car_count=2)
    absnet = AbstractRoadNetworkSystem(net, 0)
    dg_shield = compute_shield(absnet, road_safety())
    shields = {
        agent: dg_shield.with_normalizer(lambda state, a=agent: AbstractRoadNetworkSystem.normalize(state, a)) for agent in range(len(net.cars))
    }
    visualize(net, steps=0, fps=8, shields=shields)


# if __name__ == '__main__':
#     net = RoadNetwork.create_double_roundabout_scenario(car_count=2)
#     absnet = AbstractRoadNetworkSystem(net, 0)
#     safety = road_safety()
#
#     explore_queue = [(0, net.get_init_state())]
#     while explore_queue:
#         depth, state = explore_queue.pop(0)
#         safe = safety(state)
#         if safe != StateSafety.SAFE or True:
#             print(depth, safe, state)
#             safety(state)
#
#         if depth == 3 or safe == StateSafety.SAFE_RELEASE:
#             continue
#
#         actions = [2] if depth <= 1 else range(absnet.get_action_count(state))
#
#         for act in actions:
#             for succ in absnet.get_abstract_successor_states(state, act):
#                 explore_queue.append((depth + 1, succ))