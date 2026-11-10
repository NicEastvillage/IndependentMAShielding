import time
from pathlib import Path
from typing import Callable, Tuple, Generic, Optional, Any, Dict

from imashielding.shield_cache import (
    PathLike,
    ShieldCacheError,
    load_shield,
    save_shield,
    shield_cache_path,
)
from imashielding.shield_graph import (
    NodeColor,
    SAFE,
    SAFE_DIRTY,
    UNEXPLORED,
    UNSAFE,
    ShieldGraph,
    ShieldGraphBuilder,
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
    def __init__(self, graph: ShieldGraph, metadata: Optional[Dict[str, Any]] = None):
        self._graph = graph
        self.metadata = metadata or {}

    def get_safe_actions(self, state: ABS_STATE) -> Tuple[int, ...]:
        index = self._graph.index_of(state)
        if index is None:
            return ()
        # A SAFE_RELEASE state means some *other* agent already violated its
        # objective.  Under the assumption that every agent is shielded this
        # cannot happen, so being asked about such a state indicates a bug
        # (e.g. an unshielded agent, or an unsound abstraction).  Fail loudly
        # instead of silently masking the agent.
        if self._graph.released[index]:
            raise AssertionError(
                f'shield queried about release state {state!r}; a release '
                f'state is unreachable while every agent is shielded'
            )
        return self._graph.safe_actions[index]

    def contains(self, state: ABS_STATE) -> bool:
        return self._graph.contains(state)

    def iter_states(self):
        """Iterate over every abstract state represented by this shield."""
        return self._graph.iter_states()

    def is_released(self, state: ABS_STATE) -> bool:
        return self._graph.is_released(state)

    def color_of(self, state: ABS_STATE) -> Optional[NodeColor]:
        return self._graph.color_of(state)

    @property
    def state_count(self) -> int:
        return self._graph.state_count

    # ------------------------------------------------------------------ persistence

    def save(self, path: PathLike, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Write this shield to ``path`` in the versioned cache format."""
        save_shield(self._graph, path, metadata if metadata is not None else self.metadata)

    @classmethod
    def load(cls, path: PathLike) -> 'DGShield[ABS_STATE]':
        """Load a shield written by :meth:`save`.

        Raises :class:`imashielding.shield_cache.ShieldCacheError` if the file is
        missing, corrupt, or was written by an incompatible version.
        """
        graph, metadata = load_shield(path)
        return cls(graph, metadata)

    @property
    def graph(self) -> ShieldGraph:
        """The compact state/action store backing this shield."""
        return self._graph


class NormalizingShield(Shield[STATE]):
    def __init__(self, normalize: Normalization[STATE, ABS_STATE], shield: Shield[ABS_STATE]):
        self._normalize = normalize
        self._shield = shield

    def get_safe_actions(self, state: STATE) -> Tuple[int, ...]:
        return self._shield.get_safe_actions(self._normalize(state))

    def contains(self, state: STATE) -> bool:
        return self._shield.contains(self._normalize(state))


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


def _propagate_unsafe(
    builder: ShieldGraphBuilder,
    state_id: int,
    update_stack: list,
) -> None:
    """Mark all incoming actions of ``state_id`` unsafe and dirty their states.

    Every action that can reach the newly unsafe state is marked unsafe, and
    every predecessor state that is currently SAFE is queued for re-evaluation.
    """
    entries = builder.predecessors[state_id]
    action_status = builder.action_status
    colors = builder.colors
    for i in range(0, len(entries), 2):
        predecessor_id = entries[i]
        action = entries[i + 1]
        statuses = action_status[predecessor_id]
        if statuses is not None:
            statuses[action] = UNSAFE
        if colors[predecessor_id] == SAFE:
            colors[predecessor_id] = SAFE_DIRTY
            update_stack.append(predecessor_id)


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

    # Each state has one status per action; an action stays SAFE while all of
    # its successors are safe, and becomes UNSAFE as soon as one is unsafe.
    # States are coloured UNEXPLORED, SAFE, SAFE_DIRTY, or UNSAFE.  Reverse
    # dependencies are stored as (predecessor state, action) pairs so an unsafe
    # state can poison its incoming actions directly.
    # Loop invariants:
    # - If a state is coloured UNEXPLORED then it is in the explore_stack.
    # - If a state is coloured SAFE_DIRTY, then it is in the update_stack.

    PRINT_EVERY_X = 500

    print('Computing shield...')
    start_time = time.time()

    init_state = system.get_init_state()

    builder = ShieldGraphBuilder()
    init_id = builder.add_state(init_state)

    explore_stack = [init_id]
    update_stack = []

    unsafe_count = 0
    release_count = 0

    while explore_stack or update_stack:
        if update_stack:
            state_id = update_stack.pop()
        else:
            state_id = explore_stack.pop()

        color = builder.colors[state_id]
        if color == UNSAFE or color == SAFE:
            # This node is done now / already up-to-date.
            continue
        elif color == SAFE_DIRTY:
            # Check if any action is guaranteed safe. Transition to SAFE or UNSAFE accordingly.
            statuses = builder.action_status[state_id]
            any_safe = statuses is not None and any(
                status == SAFE or status == SAFE_DIRTY for status in statuses
            )
            if any_safe:
                builder.colors[state_id] = SAFE
            else:
                builder.colors[state_id] = UNSAFE
                unsafe_count += 1
                if unsafe_count % PRINT_EVERY_X == 0:
                    print('.', end='')
                _propagate_unsafe(builder, state_id, update_stack)

        elif color == UNEXPLORED:
            # Generate successors of this state. Any new states are added to the
            # graph and queued for exploration.
            state = builder.states[state_id]
            action_count = system.get_action_count(state)
            builder.begin_actions(state_id, action_count)
            statuses = builder.action_status[state_id]

            any_safe = False
            for action in range(action_count):
                action_status = SAFE
                # Deduplicate successor ids per action so reverse-dependency
                # propagation stays linear in the number of distinct successors.
                seen_successors = set()

                for successor in system.get_abstract_successor_states(state, action):
                    successor_id = builder.state_ids.get(successor)

                    if successor_id is None:
                        # New state.
                        successor_id = builder.add_state(successor)
                        seen_successors.add(successor_id)
                        builder.add_edge(successor_id, state_id, action)
                        verdict = safe_func(successor)

                        if verdict == StateSafety.UNSAFE:
                            builder.colors[successor_id] = UNSAFE
                            action_status = UNSAFE
                            break  # One bad successor is enough, skip the rest.
                        elif verdict == StateSafety.SAFE:
                            explore_stack.append(successor_id)
                        elif verdict == StateSafety.SAFE_RELEASE:
                            # No need to explore.
                            builder.colors[successor_id] = SAFE
                            builder.released[successor_id] = 1
                            release_count += 1
                        else:
                            assert False
                        continue

                    # Existing state.
                    if successor_id not in seen_successors:
                        seen_successors.add(successor_id)
                        builder.add_edge(successor_id, state_id, action)
                    if builder.colors[successor_id] == UNSAFE:
                        action_status = UNSAFE
                        break

                statuses[action] = action_status
                if action_status == SAFE:
                    any_safe = True

            if not any_safe:
                builder.colors[state_id] = UNSAFE
                unsafe_count += 1
                if unsafe_count % PRINT_EVERY_X == 0:
                    print('.', end='')
                _propagate_unsafe(builder, state_id, update_stack)
            else:
                builder.colors[state_id] = SAFE
        else:
            assert False

    end_time = time.time()
    if unsafe_count >= PRINT_EVERY_X:
        print('')

    graph, graph_nodes, graph_edges = builder.finish()
    # Release the transient action/dependency arrays before saving the shield.
    del builder

    safe_count = 0
    unsafe_count = 0
    for color in graph.colors:
        if color == SAFE:
            safe_count += 1
        elif color == UNSAFE:
            unsafe_count += 1
        else:
            assert False, f'State had color {NodeColor(color)} at the end of shielding'
    print(f'- Safe states: {safe_count}\n- Unsafe states: {unsafe_count}\n- Released states: {release_count}\n- Time (s): {end_time - start_time:0.3f}')

    metadata = {
        'cache_key': resolved_key,
        'system': type(system).__qualname__,
        'safe_func': getattr(safe_func, '__qualname__', repr(safe_func)),
        'safe_states': safe_count,
        'unsafe_states': unsafe_count,
        'release_states': release_count,
        'graph_nodes': graph_nodes,
        'graph_edges': graph_edges,
        'synthesis_seconds': end_time - start_time,
    }
    shield = DGShield(graph, metadata)

    if cache_path is not None:
        try:
            shield.save(cache_path)
            print(f'Saved shield to cache: {cache_path}')
        except OSError as error:
            # A read-only or full cache directory must not make a successful
            # synthesis fail; the shield is still usable for this run.
            print(f'Warning: could not save shield cache {cache_path}: {error}')

    return shield
