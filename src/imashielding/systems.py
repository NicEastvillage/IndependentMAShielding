from dataclasses import dataclass
from enum import Enum
from typing import Optional, TypeVar, Generic, Tuple, Callable, List

STATE = TypeVar('STATE')
ABS_STATE = TypeVar('ABS_STATE')


@dataclass(frozen=True, slots=True)
class ChancedState(Generic[STATE]):
    chance: float
    state: STATE


class ConcurrentGame(Generic[STATE]):
    def get_init_state(self) -> STATE:
        raise NotImplementedError()

    def get_successors(self, state: STATE, actions: Tuple[int, ...]) -> Tuple[ChancedState[STATE], ...]:
        raise NotImplementedError()

    def get_agent_action_count(self, state: STATE, agent: int) -> int:
        raise NotImplementedError()


class LabelledTransitionSystem(Generic[ABS_STATE]):
    def get_init_states(self) -> List[ABS_STATE]:
        raise NotImplementedError()

    def get_abstract_successor_states(self, state: ABS_STATE, action: int) -> Tuple[ABS_STATE, ...]:
        raise NotImplementedError()

    def get_action_count(self, state: ABS_STATE) -> int:
        raise NotImplementedError()

    def get_cache_key(self) -> Optional[str]:
        """A stable identifier of everything that influences the synthesized shield.

        Used to name an on-disk shield cache entry.  Implementations should return
        a string that changes whenever the transition system changes (topology,
        initial state, actions, abstraction), or ``None`` to disable automatic
        caching for this system.
        """
        return None


Normalization = Callable[[STATE], ABS_STATE]


class StateSafety(Enum):
    UNSAFE = 0
    SAFE = 1
    SAFE_RELEASE = 2
