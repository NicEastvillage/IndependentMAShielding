import argparse

from imashielding.roads import RoadNetwork
from imashielding.roads_reduction import AbstractRoadNetworkSystem, road_safety
from imashielding.shield import compute_shield
from imashielding.visualize import visualize


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resynthesize', action='store_true', help='Ignore the shield cache and synthesize again.')
    args = parser.parse_args()

    net = RoadNetwork.create_crossing_roads_scenario(car_count=3, max_vel=2)
    absnet = AbstractRoadNetworkSystem(net)
    dg_shield = compute_shield(absnet, road_safety(net), force_recompute=args.resynthesize)
    shields = {
        agent: dg_shield.with_normalizer(lambda state, a=agent: absnet.normalize(state, a)) for agent in range(len(net.cars))
    }
    visualize(net, steps=0, fps=16, shields=shields)
