"""Compact in-memory representation of a synthesized shield.

The synthesizer's dependency graph is stored with:

* states interned once and addressed by small integers,
* colours and release flags in byte arrays,
* an integer status per action instead of a separate action node,
* reverse dependencies as flat ``(predecessor, action)`` integer pairs, and
* after synthesis, only the states, colours, release flags and precomputed safe
  actions.

This keeps the synthesis fixpoint small enough to run on large abstract state
spaces while preserving the exact explored states and safe-action sets.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Generic, List, Optional, Sequence, Tuple

from imashielding.systems import ABS_STATE


class NodeColor(Enum):
    UNEXPLORED = 0
    SAFE = 1
    SAFE_DIRTY = 2
    UNSAFE = 3


# Integer aliases used in the compact arrays.  Keeping these as module constants
# avoids an enum lookup and attribute access in the hottest loop.
UNEXPLORED = NodeColor.UNEXPLORED.value
SAFE = NodeColor.SAFE.value
SAFE_DIRTY = NodeColor.SAFE_DIRTY.value
UNSAFE = NodeColor.UNSAFE.value


class ShieldGraph(Generic[ABS_STATE]):
    """The finished, read-only part of a synthesized shield.

    ``states`` maps a dense integer id to a state.  The other arrays are indexed
    by that id.  ``safe_actions[i]`` is the final shield answer for ``states[i]``;
    release states have an empty tuple there but are distinguished by
    ``released[i]`` so that querying one can fail loudly.
    """

    __slots__ = ("states", "state_ids", "colors", "released", "safe_actions")

    def __init__(
        self,
        states: Sequence[ABS_STATE],
        colors: Sequence[int],
        released: Sequence[int],
        safe_actions: Sequence[Tuple[int, ...]],
        state_ids: Optional[Dict[ABS_STATE, int]] = None,
    ):
        if not (len(states) == len(colors) == len(released) == len(safe_actions)):
            raise ValueError("compact shield arrays must all have the same length")

        self.states: List[ABS_STATE] = list(states)
        self.colors = bytearray(colors)
        self.released = bytearray(released)
        self.safe_actions: List[Tuple[int, ...]] = list(safe_actions)
        self.state_ids: Dict[ABS_STATE, int] = (
            state_ids
            if state_ids is not None
            else {state: index for index, state in enumerate(self.states)}
        )

    # ------------------------------------------------------------------ queries

    def index_of(self, state: ABS_STATE) -> Optional[int]:
        return self.state_ids.get(state)

    def contains(self, state: ABS_STATE) -> bool:
        return state in self.state_ids

    def get_safe_actions(self, state: ABS_STATE) -> Tuple[int, ...]:
        index = self.state_ids.get(state)
        if index is None:
            return ()
        return self.safe_actions[index]

    def is_released(self, state: ABS_STATE) -> bool:
        index = self.state_ids.get(state)
        return index is not None and bool(self.released[index])

    def color_of(self, state: ABS_STATE) -> Optional[NodeColor]:
        index = self.state_ids.get(state)
        if index is None:
            return None
        return NodeColor(self.colors[index])

    @property
    def state_count(self) -> int:
        return len(self.states)

    def iter_states(self):
        return iter(self.states)


class ShieldGraphBuilder:
    """Mutable compact dependency graph used during synthesis.

    States are interned to integer ids, actions are integer status entries, and
    reverse dependencies are stored as integer pairs.  ``finish`` discards the
    transient action and predecessor structures.
    """

    __slots__ = (
        "states",
        "state_ids",
        "colors",
        "released",
        "action_status",
        "predecessors",
    )

    def __init__(self):
        self.states: List[ABS_STATE] = []
        self.state_ids: Dict[ABS_STATE, int] = {}
        self.colors = bytearray()
        self.released = bytearray()
        # Per state: None until explored, then one status per action.
        self.action_status: List[Optional[List[int]]] = []
        # Per target state: flat pairs (predecessor state id, action index).
        self.predecessors: List[List[int]] = []

    def add_state(self, state: ABS_STATE) -> int:
        state_id = len(self.states)
        self.states.append(state)
        self.state_ids[state] = state_id
        self.colors.append(UNEXPLORED)
        self.released.append(0)
        self.action_status.append(None)
        self.predecessors.append([])
        return state_id

    def begin_actions(self, state_id: int, action_count: int) -> None:
        # Every action starts potentially safe.  An action becomes unsafe once
        # any of its successors is (or becomes) unsafe.
        self.action_status[state_id] = [SAFE] * action_count

    def add_edge(self, target_id: int, source_id: int, action: int) -> None:
        entries = self.predecessors[target_id]
        entries.append(source_id)
        entries.append(action)

    def finish(self) -> tuple[ShieldGraph[ABS_STATE], int, int]:
        """Return the compact shield and logical dependency-graph node/edge counts."""
        safe_actions: List[Tuple[int, ...]] = []
        action_nodes = 0
        for state_id, statuses in enumerate(self.action_status):
            if statuses is None:
                # Never explored: an outright unsafe or a release state.
                safe_actions.append(())
                continue
            action_nodes += len(statuses)
            if self.colors[state_id] == SAFE:
                safe_actions.append(
                    tuple(action for action, status in enumerate(statuses) if status == SAFE)
                )
            else:
                safe_actions.append(())

        edge_count = action_nodes
        for entries in self.predecessors:
            edge_count += len(entries) // 2

        graph = ShieldGraph(
            self.states,
            self.colors,
            self.released,
            safe_actions,
            state_ids=self.state_ids,
        )
        return graph, len(self.states) + action_nodes, edge_count
