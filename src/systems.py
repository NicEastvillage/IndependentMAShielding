from typing import TypeVar, Generic, List, Tuple

STATE = TypeVar('STATE')

class LabelledTransitionSystem(Generic[STATE]):
    def get_init_state(self) -> STATE:
        raise NotImplementedError()

    def get_abstract_successor_states(self, state: STATE, action: int) -> Tuple[STATE, ...]:
        raise NotImplementedError()

    def get_agent_action_count(self, state: STATE) -> int:
        raise NotImplementedError()
