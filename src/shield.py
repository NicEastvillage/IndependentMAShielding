from typing import Callable, Tuple, Generic

from networkx import DiGraph

from systems import LabelledTransitionSystem, STATE


class Shield(Generic[STATE]):
    def __init__(self, system: LabelledTransitionSystem[STATE], agent: int, graph: DiGraph):
        self._system = system
        self._agent = agent
        self._graph = graph

    def get_safe_actions(self, state: STATE) -> Tuple[int, ...]:
        state = self._system._normalize(state, self._agent)   # FIXME: Not generic
        return tuple(self._graph.nodes[act_node]['action'] for act_node in self._graph.successors(state) if self._graph.nodes[act_node]['color'] == SAFE)

    def symmetric_for(self, agent: int):
        return Shield(self._system, agent, self._graph)


UNEXPLORED = 0
SAFE = 1
SAFE_DIRTY = 2
UNSAFE = 3


def compute_shield(system: LabelledTransitionSystem[STATE], agent: int, safe_func: Callable) -> Shield[STATE]:
    # The shield is computed by constructing an abstract dependency graph.
    # This implementation uses a bi-partite graph: Both states and actions appear are nodes.
    # State nodes are marked with a color: UNEXPLORED, SAFE, SAFE_DIRTY, or UNSAFE.
    # Action nodes are marked with a color: SAFE or UNSAFE.
    # Loop invariants:
    # - If a node is colored UNEXPLORED then it is in the explore_stack.
    # - if a node is colored SAFE_DIRTY, then it is in the update_stack.

    init_state = system.get_init_state()

    g = DiGraph()
    g.add_node(init_state, color=UNEXPLORED)

    explore_stack = [init_state]
    update_stack = []

    counter = 0

    while len(explore_stack) > 0 or len(update_stack) > 0:
        counter += 1
        print("counter:", counter)
        if len(update_stack) > 0:
            n = update_stack.pop()
        else:
            n = explore_stack.pop()
        color = g.nodes[n]['color']
        if color == UNSAFE:
            # This node is done now
            continue
        elif color == SAFE:
            # This node is already up-to-date
            continue
        elif color == SAFE_DIRTY:
            # Check if any action is guaranteed safe. Transition to SAFE or UNSAFE accordingly.
            any_safe = False
            for act in range(system.get_agent_action_count(n)):
                action_node = (n, act)
                if g.nodes[action_node]['color'] == SAFE or g.nodes[action_node]['color'] == SAFE_DIRTY:
                    any_safe = True
            if any_safe:
                g.nodes[n]['color'] = SAFE
            else:
                g.nodes[n]['color'] = UNSAFE
                # Queue SAFE predecessors as SAFE_DIRTY
                for pred_act in g.predecessors(n):
                    g.nodes[pred_act]['color'] = UNSAFE
                    for pred in g.predecessors(pred_act):
                        if g.nodes[pred]['color'] == SAFE:
                            g.nodes[pred]['color'] = SAFE_DIRTY
                            update_stack.append(pred)

        elif color == UNEXPLORED:
            # Generate successors of this state. Any new states are added to the graph and queued for exploration
            any_safe = False
            for act in range(system.get_agent_action_count(n)):
                action_node = (n, act)
                g.add_node(action_node, color=SAFE, action=act)
                g.add_edge(n, action_node)
                for succ in system.get_abstract_successor_states(n, act):
                    if succ in g.nodes:
                        # Existing
                        g.add_edge(action_node, succ)
                        if g.nodes[succ]['color'] == UNSAFE:
                            g.nodes[action_node]['color'] = UNSAFE
                            break  # We only need one bad action successor for unsafety, skip the rest
                    else:
                        # New
                        g.add_node(succ, color=UNEXPLORED)
                        g.add_edge(action_node, succ)

                        if not safe_func(succ):
                            g.nodes[succ]['color'] = UNSAFE
                            g.nodes[action_node]['color'] = UNSAFE
                            break  # We only need one bad action successor for unsafety, skip the rest

                        explore_stack.append(succ)

                if g.nodes[action_node]['color'] == SAFE:
                    any_safe = True

            if not any_safe:
                g.nodes[n]['color'] = UNSAFE
                # Conclusion reached already, queue predecessors for update
                for pred_act in g.predecessors(n):
                    g.nodes[pred_act]['color'] = UNSAFE
                    for pred in g.predecessors(pred_act):
                        if g.nodes[pred]['color'] == SAFE:
                            g.nodes[pred]['color'] = SAFE_DIRTY
                            update_stack.append(pred)
            else:
                g.nodes[n]['color'] = SAFE
        else:
            assert False

    safe_count, unsafe_count = 0, 0
    for n in g.nodes:
        if g.nodes[n]['color'] == SAFE:
            safe_count += 1
        elif g.nodes[n]['color'] == UNSAFE:
            unsafe_count += 1
        else:
            assert False, f'Node had color {g.nodes[n]['color']} at the end of shielding'
    print(safe_count, unsafe_count)

    return Shield(system, agent, g)
