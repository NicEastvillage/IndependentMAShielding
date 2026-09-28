import time
from enum import Enum
from typing import Callable, Tuple, Generic

from networkx import DiGraph

from imashielding.systems import LabelledTransitionSystem, ABS_STATE, STATE, Normalization, StateSafety


class Shield(Generic[ABS_STATE]):
    def get_safe_actions(self, state: ABS_STATE) -> Tuple[int, ...]:
        raise NotImplementedError()

    def with_normalizer(self, normalizer: Normalization[STATE, ABS_STATE]) -> 'Shield[STATE]':
        return NormalizingShield(normalizer, self)


class DGShield(Shield[ABS_STATE]):
    def __init__(self, graph: DiGraph):
        self._graph = graph

    def get_safe_actions(self, state: ABS_STATE) -> Tuple[int, ...]:
        return tuple(self._graph.nodes[act_node]['action']
                     for act_node in self._graph.successors(state)
                     if self._graph.nodes[act_node]['color'] == NodeColor.SAFE)


class NormalizingShield(Shield[STATE]):
    def __init__(self, normalize: Normalization[STATE, ABS_STATE], shield: Shield[ABS_STATE]):
        self._normalize = normalize
        self._shield = shield

    def get_safe_actions(self, state: ABS_STATE) -> Tuple[int, ...]:
        return self._shield.get_safe_actions(self._normalize(state))


class NodeColor(Enum):
    UNEXPLORED = 0
    SAFE = 1
    SAFE_DIRTY = 2
    UNSAFE = 3


def compute_shield(system: LabelledTransitionSystem[ABS_STATE], safe_func: Callable[[ABS_STATE], StateSafety]) -> DGShield[STATE]:
    # The shield is computed by constructing an abstract dependency graph.
    # This implementation uses a bi-partite graph: Both states and actions appear are nodes.
    # State nodes are marked with a color: UNEXPLORED, SAFE, SAFE_DIRTY, or UNSAFE.
    # Action nodes are marked with a color: SAFE or UNSAFE.
    # Loop invariants:
    # - If a node is colored UNEXPLORED then it is in the explore_stack.
    # - if a node is colored SAFE_DIRTY, then it is in the update_stack.

    PRINT_EVERY_X = 500

    print('Computing shield...')
    start_time = time.time()

    init_state = system.get_init_state()

    g = DiGraph()
    g.add_node(init_state, color=NodeColor.UNEXPLORED)

    explore_stack = [init_state]
    update_stack = []

    unsafe_count = 0
    release_count = 0

    while len(explore_stack) > 0 or len(update_stack) > 0:
        if len(update_stack) > 0:
            n = update_stack.pop()
        else:
            n = explore_stack.pop()
        color = g.nodes[n]['color']
        if color == NodeColor.UNSAFE:
            # This node is done now
            continue
        elif color == NodeColor.SAFE:
            # This node is already up-to-date
            continue
        elif color == NodeColor.SAFE_DIRTY:
            # Check if any action is guaranteed safe. Transition to SAFE or UNSAFE accordingly.
            any_safe = False
            for act in range(system.get_action_count(n)):
                action_node = (n, act)
                if g.nodes[action_node]['color'] == NodeColor.SAFE or g.nodes[action_node]['color'] == NodeColor.SAFE_DIRTY:
                    any_safe = True
            if any_safe:
                g.nodes[n]['color'] = NodeColor.SAFE
            else:
                g.nodes[n]['color'] = NodeColor.UNSAFE
                unsafe_count += 1
                if unsafe_count % PRINT_EVERY_X == 0:
                    print('.', end='')
                # Queue SAFE predecessors as SAFE_DIRTY
                for pred_act in g.predecessors(n):
                    g.nodes[pred_act]['color'] = NodeColor.UNSAFE
                    for pred in g.predecessors(pred_act):
                        if g.nodes[pred]['color'] == NodeColor.SAFE:
                            g.nodes[pred]['color'] = NodeColor.SAFE_DIRTY
                            update_stack.append(pred)

        elif color == NodeColor.UNEXPLORED:
            # Generate successors of this state. Any new states are added to the graph and queued for exploration
            any_safe = False
            for act in range(system.get_action_count(n)):
                action_node = (n, act)
                g.add_node(action_node, color=NodeColor.SAFE, action=act)
                g.add_edge(n, action_node)
                for succ in system.get_abstract_successor_states(n, act):
                    if succ in g.nodes:
                        # Existing
                        g.add_edge(action_node, succ)
                        if g.nodes[succ]['color'] == NodeColor.UNSAFE:
                            g.nodes[action_node]['color'] = NodeColor.UNSAFE
                            break  # We only need one bad action successor for unsafety, skip the rest
                    else:
                        # New
                        g.add_node(succ, color=NodeColor.UNEXPLORED)
                        g.add_edge(action_node, succ)
                        verdict = safe_func(succ)

                        if verdict == StateSafety.UNSAFE:
                            g.nodes[succ]['color'] = NodeColor.UNSAFE
                            g.nodes[action_node]['color'] = NodeColor.UNSAFE
                            break  # We only need one bad action successor for unsafety, skip the rest

                        elif verdict == StateSafety.SAFE:
                            explore_stack.append(succ)

                        elif verdict == StateSafety.SAFE_RELEASE:
                            # No need to explore
                            g.nodes[succ]['color'] = NodeColor.SAFE
                            release_count += 1

                        else:
                            assert False

                if g.nodes[action_node]['color'] == NodeColor.SAFE:
                    any_safe = True

            if not any_safe:
                g.nodes[n]['color'] = NodeColor.UNSAFE
                unsafe_count += 1
                if unsafe_count % PRINT_EVERY_X == 0:
                    print('.', end='')
                # Conclusion reached already, queue predecessors for update
                for pred_act in g.predecessors(n):
                    g.nodes[pred_act]['color'] = NodeColor.UNSAFE
                    for pred in g.predecessors(pred_act):
                        if g.nodes[pred]['color'] == NodeColor.SAFE:
                            g.nodes[pred]['color'] = NodeColor.SAFE_DIRTY
                            update_stack.append(pred)
            else:
                g.nodes[n]['color'] = NodeColor.SAFE
        else:
            assert False

    end_time = time.time()
    if unsafe_count >= PRINT_EVERY_X:
        print('')

    safe_count, unsafe_count = 0, 0
    for n in g.nodes:
        if 'action' in g.nodes[n]:
            continue   # An action node
        if g.nodes[n]['color'] == NodeColor.SAFE:
            safe_count += 1
        elif g.nodes[n]['color'] == NodeColor.UNSAFE:
            unsafe_count += 1
        else:
            assert False, f'Node had color {g.nodes[n]['color']} at the end of shielding'
    print(f'- Safe states: {safe_count}\n- Unsafe states: {unsafe_count}\n- Released states: {release_count}\n- Time (s): {end_time - start_time:0.3f}')

    return DGShield(g)
