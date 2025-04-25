from typing import Callable
from typeguard import typechecked

@typechecked
class RulesEngine:
    def __init__(self) -> None:
        # Each rule is a dict with keys "conditions" and "actions"
        # "conditions": list of callables returning bool
        # "actions": list of callables returning None
        self.rules: list[dict[str, list[Callable[[], bool]] | list[Callable[[], None]]]] = []

    def add_rule(
        self,
        conditions: Callable[[], bool] | list[Callable[[], bool]],
        actions: Callable[[], None] | list[Callable[[], None]]
    ) -> None:
        if not isinstance(conditions, list):
            conditions = [conditions]
        if not isinstance(actions, list):
            actions = [actions]

        # We know conditions is list[Callable[[], bool]] and actions is list[Callable[[], None]]
        # But because Python's union in dict value types can't enforce this precisely,
        # the following type fix is fine for our usage.
        self.rules.append({
            "conditions": conditions,  # type: ignore[assignment]
            "actions": actions  # type: ignore[assignment]
        })

    def evaluate_rules(self) -> None:
        for rule in self.rules:
            # Force typing here to satisfy type checkers
            conditions = rule["conditions"]  # type: list[Callable[[], bool]]
            if all(condition() for condition in conditions):
                actions = rule["actions"]  # type: list[Callable[[], None]]
                for action in actions:
                    action()