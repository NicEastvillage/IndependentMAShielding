from roads import RoadNetwork
from roads_reduction import AbstractRoundaboutSystem, roundabout_safety
from shield import compute_shield
from visualize import visualize


if __name__ == '__main__':
    net = RoadNetwork.create_roundabout_scenario(road_length=100)
    absnet = AbstractRoundaboutSystem(net, 0)
    shield = compute_shield(absnet, 0, roundabout_safety(4))
    visualize(net, steps=0, shields={agent: shield.symmetric_for(agent) for agent in range(len(net.cars))})
