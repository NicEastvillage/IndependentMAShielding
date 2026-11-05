import time
from enum import Enum
from pathlib import Path
from typing import Callable, Tuple, Generic, Optional, Any, Dict

from networkx import DiGraph

from imashielding.shield_cache import (
    PathLike,
    ShieldCacheError,
    load_graph,
    save_graph,
    shield_cache_path,
)
from imashielding.systems import LabelledTransitionSystem, ABS_STATE, STATE, Normalization, StateSafety


class Shield(Generic[ABS_STATE]):
    def get_safe_actions(self, state: ABS_STATE) -> Tuple[int, ...]:
        raise NotImplementedError()

    def contains(self, state: ABS_STATE) -> bool:
        """Whether this shield has any information about `state`.

        A shield synthesized on-the-fly only covers the states it actually reached during
        synthesis. States outside that set are outside the shield's scope.
        """
        raise NotImplementedError()

    def with_normalizer(self, normalizer: Normalization[STATE, ABS_STATE]) -> 'Shield[STATE]':
        return NormalizingShield(normalizer, self)


class DGShield(Shield[ABS_STATE]):
    def __init__(self, graph: DiGraph, metadata: Optional[Dict[str, Any]] = None):
        self._graph = graph
        self.metadata = metadata or {}

    def get_safe_actions(self, state: ABS_STATE) -> Tuple[int, ...]:
        # A SAFE_RELEASE state means some *other* agent already violated its
        # objective.  Under the assumption that every agent is shielded this
        # cannot happen, so being asked about such a state indicates a bug
        # (e.g. an unshielded agent, or an unsound abstraction).  Fail loudly
        # instead of silently masking the agent.
        if state in self._graph.nodes:
            data = self._graph.nodes[state]
            released = data.get('released', False) or (
                data.get('color') == NodeColor.SAFE
                and self._graph.out_degree(state) == 0
            )
            if released:
                raise AssertionError(
                    f'shield queried about release state {state!r}; a release '
                    f'state is unreachable while every agent is shielded'
                )
        return tuple(self._graph.nodes[act_node]['action']
                     for act_node in self._graph.successors(state)
                     if self._graph.nodes[act_node]['color'] == NodeColor.SAFE)

    def contains(self, state: ABS_STATE) -> bool:
        return state in self._graph.nodes

    # ------------------------------------------------------------------ persistence

    def save(self, path: PathLike, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Write this shield to ``path`` in the versioned cache format."""
        save_graph(self._graph, path, metadata if metadata is not None else self.metadata)

    @classmethod
    def load(cls, path: PathLike) -> 'DGShield[ABS_STATE]':
        """Load a shield written by :meth:`save`.

        Raises :class:`imashielding.shield_cache.ShieldCacheError` if the file is
        missing, corrupt, or was written by an incompatible version.
        """
        graph, metadata = load_graph(path)
        return cls(graph, metadata)

    @property
    def graph(self) -> DiGraph:
        return self._graph


class NormalizingShield(Shield[STATE]):
    def __init__(self, normalize: Normalization[STATE, ABS_STATE], shield: Shield[ABS_STATE]):
        self._normalize = normalize
        self._shield = shield

    def get_safe_actions(self, state: STATE) -> Tuple[int, ...]:
        return self._shield.get_safe_actions(self._normalize(state))

    def contains(self, state: STATE) -> bool:
        return self._shield.contains(self._normalize(state))


class NodeColor(Enum):
    UNEXPLORED = 0
    SAFE = 1
    SAFE_DIRTY = 2
    UNSAFE = 3


def _resolve_cache_key(
    system: LabelledTransitionSystem[ABS_STATE],
    safe_func: Callable[[ABS_STATE], StateSafety],
    cache_key: Optional[str],
) -> Optional[str]:
    """Derive a cache key from the system and the safety property.

    Automatic caching requires both halves to identify themselves: a system via
    :meth:`LabelledTransitionSystem.get_cache_key` and a safety predicate via a
    ``shield_cache_key`` attribute (set by the safety-property factories).  A
    caller-provided key always wins and also works for custom objects.
    """
    if cache_key is not None:
        return cache_key
    system_key_func = getattr(system, 'get_cache_key', None)
    system_key = system_key_func() if callable(system_key_func) else None
    safe_key = getattr(safe_func, 'shield_cache_key', None)
    if callable(safe_key):
        safe_key = safe_key()
    if system_key is None or safe_key is None:
        return None
    return f'{system_key}|{safe_key}'


def compute_shield(
    system: LabelledTransitionSystem[ABS_STATE],
    safe_func: Callable[[ABS_STATE], StateSafety],
    *,
    cache_key: Optional[str] = None,
    cache_dir: Optional[PathLike] = None,
    force_recompute: bool = False,
) -> DGShield[ABS_STATE]:
    """Synthesize the most permissive shield for ``system`` and ``safe_func``.

    When both the system and the safety property provide a cache key (the road
    systems and safety properties in this package do), the synthesized shield is
    automatically saved to the shield cache.  If a compatible cached shield
    already exists it is loaded instead of being synthesized again.  Pass
    ``force_recompute=True`` to specifically request a fresh synthesis, or
    ``cache_key``/``cache_dir`` to control where the result is stored.
    """
    resolved_key = _resolve_cache_key(system, safe_func, cache_key)
    cache_path: Optional[Path] = (
        shield_cache_path(resolved_key, cache_dir) if resolved_key is not None else None
    )

    if cache_path is not None and not force_recompute:
        if cache_path.is_file():
            try:
                shield = DGShield.load(cache_path)
                init_state = system.get_init_state()
                if not shield.contains(init_state):
                    raise ShieldCacheError(
                        f'the cached shield does not cover the initial state {init_state!r}'
                    )
                print(f'Loaded shield from cache: {cache_path}')
                return shield
            except ShieldCacheError as error:
                print(f'Ignoring unusable shield cache ({error}); synthesizing again.')

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
                            g.nodes[succ]['released'] = True
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

    metadata = {
        'cache_key': resolved_key,
        'system': type(system).__qualname__,
        'safe_func': getattr(safe_func, '__qualname__', repr(safe_func)),
        'safe_states': safe_count,
        'unsafe_states': unsafe_count,
        'release_states': release_count,
        'graph_nodes': g.number_of_nodes(),
        'graph_edges': g.number_of_edges(),
        'synthesis_seconds': end_time - start_time,
    }
    shield = DGShield(g, metadata)

    if cache_path is not None:
        try:
            shield.save(cache_path)
            print(f'Saved shield to cache: {cache_path}')
        except OSError as error:
            # A read-only or full cache directory must not make a successful
            # synthesis fail; the shield is still usable for this run.
            print(f'Warning: could not save shield cache {cache_path}: {error}')

    return shield
