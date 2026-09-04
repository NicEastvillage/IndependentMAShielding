import colorsys
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Mapping

import pygame

from roads import RoadNetwork, RoadNetworkState
from shield import Shield

WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 720

ROAD_WIDTH = 0.17
CAR_LENGTH = 0.15
CAR_WIDTH = 0.09
WINDSHIELD_HALF_WIDTH = CAR_WIDTH * 0.42
WINDSHIELD_FRONT = CAR_LENGTH * 0.5 * 0.55
WINDSHIELD_BACK = CAR_LENGTH * 0.5 * 0.15
HEADLIGHT_FRONT = CAR_LENGTH * 0.5 * 0.9
HEADLIGHT_LATERAL = CAR_WIDTH * 0.5 * 0.8
HEADLIGHT_RADIUS = 0.012
BRAKE_LATERAL = CAR_WIDTH * 0.5 * 0.8
BRAKE_REAR = CAR_LENGTH * 0.5 * 0.9
BRAKE_RADIUS = 0.01

BACKGROUND = (16, 18, 24)
ASPHALT_COLOR = (55, 58, 66)
WINDSHIELD_COLOR = (140, 190, 235)
HEADLIGHT_COLOR = (255, 236, 160)
BRAKE_COLOR = (220, 30, 30)
TEXT_COLOR = (15, 15, 15)


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
            pygame.draw.circle(surface, ASPHALT_COLOR, center, int(radius + width / 2), int(width))
        else:
            # pygame.draw.arc grows the width inward, so pad the rect to center the band
            diameter = max(int(2 * radius + width), 1)
            rect = pygame.Rect(0, 0, diameter, diameter)
            rect.center = center
            start = arc.start % math.tau   # pygame's arc angles share the world convention
            pygame.draw.arc(surface, ASPHALT_COLOR, rect, start, start + abs(arc.sweep), int(width))


def draw_cars(surface: pygame.Surface, state: RoadNetworkState, view: View, font: pygame.font.Font):
    n = len(state.cars)
    for car in state.cars:
        segment = state.system.roads[car.road]
        pos, heading = segment.arc.frame(car.pos / segment.length)

        fx, fy = math.cos(heading), math.sin(heading)
        nx, ny = -fy, fx
        hx, hy = fx * CAR_LENGTH * 0.5, fy * CAR_LENGTH * 0.5
        vx, vy = nx * CAR_WIDTH * 0.5, ny * CAR_WIDTH * 0.5

        # Body
        corners = [(pos[0] + hx + vx, pos[1] + hy + vy), (pos[0] - hx + vx, pos[1] - hy + vy),
                   (pos[0] - hx - vx, pos[1] - hy - vy), (pos[0] + hx - vx, pos[1] + hy - vy)]
        pygame.draw.polygon(surface, car_color(car.id, n), [view.point(c) for c in corners])

        # Windshield
        ws = [(pos[0] + fx * WINDSHIELD_FRONT + nx * WINDSHIELD_HALF_WIDTH, pos[1] + fy * WINDSHIELD_FRONT + ny * WINDSHIELD_HALF_WIDTH),
              (pos[0] + fx * WINDSHIELD_FRONT - nx * WINDSHIELD_HALF_WIDTH, pos[1] + fy * WINDSHIELD_FRONT - ny * WINDSHIELD_HALF_WIDTH),
              (pos[0] + fx * WINDSHIELD_BACK - nx * WINDSHIELD_HALF_WIDTH, pos[1] + fy * WINDSHIELD_BACK - ny * WINDSHIELD_HALF_WIDTH),
              (pos[0] + fx * WINDSHIELD_BACK + nx * WINDSHIELD_HALF_WIDTH, pos[1] + fy * WINDSHIELD_BACK + ny * WINDSHIELD_HALF_WIDTH)]
        pygame.draw.polygon(surface, WINDSHIELD_COLOR, [view.point(c) for c in ws])

        # Headlights
        for side in (HEADLIGHT_LATERAL, -HEADLIGHT_LATERAL):
            light = (pos[0] + fx * HEADLIGHT_FRONT + nx * side, pos[1] + fy * HEADLIGHT_FRONT + ny * side)
            pygame.draw.circle(surface, HEADLIGHT_COLOR, view.point(light), max(int(view.length(HEADLIGHT_RADIUS)), 1))

        # Brake lights
        if car.braking:
            for side in (BRAKE_LATERAL, -BRAKE_LATERAL):
                rear = (pos[0] - fx * BRAKE_REAR + nx * side, pos[1] - fy * BRAKE_REAR + ny * side)
                pygame.draw.circle(surface, BRAKE_COLOR, view.point(rear), max(int(view.length(BRAKE_RADIUS)), 1))

        label = font.render(str(car.vel), True, TEXT_COLOR)
        surface.blit(label, label.get_rect(center=view.point(pos)))


def visualize(net: RoadNetwork, steps: Optional[int] = None, shields: Mapping[int, Shield[RoadNetworkState]] = {}, fps: int = 8):
    state = net.get_init_state()

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Roundabout scenario")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    view = create_view(net, WINDOW_WIDTH, WINDOW_HEIGHT)

    paused = steps == 0
    step = 0
    while True:
        manual_step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_e:
                    manual_step = True
                    paused = True

        if not paused or manual_step:

            actions = tuple(random.randrange(net.get_agent_action_count(state, c.id)) if c.id not in shields else random.choice(shields[c.id].get_safe_actions(state)) for c in net.cars)
            successors = state.get_successors(actions)
            state = random.choices(successors, weights=[s.chance for s in successors], k=1)[0].state

            step += 1
            if step == steps:
                paused = True

        screen.fill(BACKGROUND)
        draw_roads(screen, net, view)
        draw_cars(screen, state, view, font)
        pygame.display.flip()
        clock.tick(fps)
