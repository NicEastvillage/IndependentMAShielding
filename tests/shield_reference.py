"""Independent reference implementation of shield synthesis, for tests only.

The production synthesizer in :mod:`imashielding.shield` is an on-the-fly
abstract dependency graph (ADG) fixpoint.  To guard it against logical
regressions while it is being optimised, the tests compare it against this
deliberately simple, non-incremental implementation:

1. Explore the reachable abstract transition system breadth-first.
2. Take the greatest fixpoint of the set of states that have at least one
   action whose *every* successor is still in the set.
3. A state's shield is all actions whose successors are in the fixpoint.

This mirrors the definition in the paper, not the implementation, so it is slow
and stores everything explicitly.  It is only used with small systems.

The assertions only use the public shield API (``contains`` and
``get_safe_actions``), so the tests remain valid if the internal representation
is replaced during optimisation.
"""

from collections import deque
from typing import Dict, FrozenSet, Set, Tuple

from imashielding.systems import ABS_STATE, LabelledTransitionSystem, StateSafety

# (state, action) -> nondeterministic successor set
SuccessorMap = Dict[Tuple[ABS_STATE, int], FrozenSet[ABS_STATE]]


def reference_shield(
    system: LabelledTransitionSystem[ABS_STATE],
    safe_func,
) -> Tuple[Dict[ABS_STATE, StateSafety], SuccessorMap, Dict[ABS_STATE, Tuple[int, ...]]]:
    """Compute (state verdicts, all explored transitions, safe actions).

    Release states are terminal: they get a verdict but are not expanded, and
    the returned safe-action map has no entry for them.  The initial state is
    assumed to be safe, which every scenario used by the tests satisfies.
    """
    init = system.get_init_state()
    verdict: Dict[ABS_STATE, StateSafety] = {init: safe_func(init)}
    if verdict[init] is not StateSafety.SAFE:
        raise AssertionError("the reference oracle expects a SAFE initial state")

    successors: SuccessorMap = {}
    queue = deque([init])
    while queue:
        state = queue.popleft()
        if verdict[state] is not StateSafety.SAFE:
            continue
        for action in range(system.get_action_count(state)):
            action_successors = frozenset(
                system.get_abstract_successor_states(state, action)
            )
            successors[(state, action)] = action_successors
            for successor in action_successors:
                if successor not in verdict:
                    verdict[successor] = safe_func(successor)
                    queue.append(successor)

    # Greatest fixpoint: start with every state except the outright unsafe
    # ones, then repeatedly remove states no action can keep inside the set.
    winning: Set[ABS_STATE] = {
        state for state, value in verdict.items() if value is not StateSafety.UNSAFE
    }
    changed = True
    while changed:
        changed = False
        for state in tuple(winning):
            if verdict[state] is not StateSafety.SAFE:
                continue
            if not any(
                successors[(state, action)] <= winning
                for action in range(system.get_action_count(state))
            ):
                winning.discard(state)
                changed = True

    safe_actions = {
        state: tuple(
            action
            for action in range(system.get_action_count(state))
            if successors[(state, action)] <= winning
        )
        for state, value in verdict.items()
        if value is StateSafety.SAFE
    }
    return verdict, successors, safe_actions


def assert_matches_reference(test_case, system, safe_func, shield) -> None:
    """Assert that ``shield`` agrees with the reference oracle on every state.

    States the reference can reach but the shield never generated are allowed
    (the ADG short-circuits an unsafe action, so it may not enumerate all of
    that action's successors).  Every state the shield *does* know about must
    have exactly the reference's safe-action set.
    """
    verdict, successors, expected = reference_shield(system, safe_func)

    checked = 0
    for state, value in verdict.items():
        if not shield.contains(state):
            continue
        checked += 1
        if value is StateSafety.SAFE:
            test_case.assertEqual(
                shield.get_safe_actions(state),
                expected[state],
                msg=f"safe actions differ for {state!r}",
            )
            # Soundness: every successor of an action the shield allows must
            # itself be represented in the shield (otherwise a later lookup
            # would fall outside the synthesized graph).
            for action in expected[state]:
                for successor in successors[(state, action)]:
                    test_case.assertTrue(
                        shield.contains(successor),
                        msg=f"safe action {action} of {state!r} reaches "
                        f"{successor!r}, which is missing from the shield",
                    )
        else:
            # UNSAFE: no action is allowed.  SAFE_RELEASE: terminal, and a
            # query on it is a bug (it cannot be reached while all agents are
            # shielded), so the shield must raise rather than return an empty
            # action set.
            if value is StateSafety.SAFE_RELEASE:
                with test_case.assertRaises(
                    AssertionError,
                    msg=f"expected an assertion for release state {state!r}",
                ):
                    shield.get_safe_actions(state)
            else:
                test_case.assertEqual(
                    shield.get_safe_actions(state),
                    (),
                    msg=f"expected no safe actions for unsafe state {state!r}",
                )

    test_case.assertGreater(checked, 0, msg="reference and shield share no states")
