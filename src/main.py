from roads import RoadNetwork
from visualize import visualize

if __name__ == '__main__':
    net = RoadNetwork.create_double_roundabout_scenario(road_per_roundabout=60)
    visualize(net)
