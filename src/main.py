from roads import RoadNetwork
from roads_reduction import AbstractRoundaboutSystem, roundabout_safety
from shield import compute_shield
from visualize import visualize


if __name__ == '__main__':
    net = RoadNetwork.create_roundabout_scenario(road_length=200)
    absnet = AbstractRoundaboutSystem(net, 0)
    dg_shield = compute_shield(absnet, roundabout_safety(6))
    shields = {
        agent: dg_shield.with_normalizer(lambda state, a=agent: AbstractRoundaboutSystem.normalize(state, a)) for agent in range(len(net.cars))
    }
    visualize(net, steps=0, shields=shields, fps=24)
