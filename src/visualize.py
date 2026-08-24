import colorsys
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import pygame

from roads import RoadNetwork, RoadNetworkState

WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 720
FPS = 8

ROAD_WIDTH = 0.17
CAR_LENGTH = 0.15
CAR_WIDTH = 0.09

BACKGROUND = (16, 18, 24)
ASPHALT = (55, 58, 66)
TEXT = (15, 15, 15)


@dataclass(frozen=True)
class View:
    """World-to-screen mapping fitted to the window (y-flipped)."""
    window_center: Tuple[float, float]
    world_center: Tuple[float, float]
    scale: float

    def point(self, p: Tuple[float, float]) -> Tuple[float, float]:
        return (self.window_center[0] + (p[0] - self.world_center[0]) * self.scale,
                self.window_center[1] - (p[1] - self.world_center[1]) * self.scale)

    def length(self, d: float) -> float:
        return d * self.scale


def create_view(net: RoadNetwork, window_width: int, window_height: int) -> View:
    xs, ys = [], []
    for segment in net.roads:
        for i in range(65):
            (x, y), _ = segment.arc.frame(i / 64)
            xs.append(x)
            ys.append(y)

    margin = window_height * 0.08
    span_hori = max(xs) - min(xs)
    span_vert = max(ys) - min(ys)
    scale_hori = (window_width - 2 * margin) / span_hori
    scale_vert = (window_height - 2 * margin) / span_vert
    return View(
        (window_width / 2, window_height / 2),
        ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2),
        min(scale_hori, scale_vert)
    )


def car_color(i: int, n: int) -> Tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_roads(surface: pygame.Surface, net: RoadNetwork, view: View):
    for segment in net.roads:
        arc = segment.arc
        width = view.length(ROAD_WIDTH)

        center = view.point(arc.center)
        radius = view.length(arc.radius)

        if abs(arc.sweep) >= math.tau - 1e-9:   # closed loop: draw as ring to avoid the arc seam
            pygame.draw.circle(surface, ASPHALT, center, int(radius + width / 2), int(width))
        else:
            # pygame.draw.arc grows the width inward, so pad the rect to center the band
            diameter = max(int(2 * radius + width), 1)
            rect = pygame.Rect(0, 0, diameter, diameter)
            rect.center = center
            start = arc.start % math.tau   # pygame's arc angles share the world convention
            pygame.draw.arc(surface, ASPHALT, rect, start, start + abs(arc.sweep), int(width))


def draw_cars(surface: pygame.Surface, state: RoadNetworkState, view: View, font: pygame.font.Font):
    n = len(state.cars)
    for car in state.cars:
        segment = state.system.roads[car.road]
        pos, heading = segment.arc.frame(car.pos / segment.length)

        fx, fy = math.cos(heading), math.sin(heading)
        nx, ny = -fy, fx
        hx, hy = fx * CAR_LENGTH * 0.5, fy * CAR_LENGTH * 0.5
        vx, vy = nx * CAR_WIDTH * 0.5, ny * CAR_WIDTH * 0.5
        corners = [(pos[0] + hx + vx, pos[1] + hy + vy), (pos[0] - hx + vx, pos[1] - hy + vy),
                   (pos[0] - hx - vx, pos[1] - hy - vy), (pos[0] + hx - vx, pos[1] + hy - vy)]
        pygame.draw.polygon(surface, car_color(car.id, n), [view.point(c) for c in corners])

        label = font.render(str(car.vel), True, TEXT)
        surface.blit(label, label.get_rect(center=view.point(pos)))


def advance(net: RoadNetwork, state: RoadNetworkState) -> RoadNetworkState:
    actions = [random.randrange(net.get_agent_action_count(state)) for _ in net.cars]
    successors = state.get_successors(actions)
    return random.choices(successors, weights=[s.chance for s in successors], k=1)[0].state


def main(steps: Optional[int] = None):
    #net = RoadNetwork.create_roundabout_scenario(road_length=100)
    net = RoadNetwork.create_double_roundabout_scenario()
    state = net.get_init_state()

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Roundabout scenario")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    view = create_view(net, WINDOW_WIDTH, WINDOW_HEIGHT)

    paused = False
    step = 0
    while steps is None or step < steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_RIGHT and paused:
                    state = advance(net, state)
                    step += 1

        if not paused:
            state = advance(net, state)
            step += 1

        screen.fill(BACKGROUND)
        draw_roads(screen, net, view)
        draw_cars(screen, state, view, font)
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
