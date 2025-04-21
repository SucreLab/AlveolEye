from typing import Callable, List, Dict, Union
from typeguard import typechecked


@typechecked
class RulesEngine:
    def __init__(self) -> None:
        self.rules: List[Dict[str, List[Callable[[], None]]]] = []

    def add_rule(
        self,
        conditions: Union[Callable[[], bool], List[Callable[[], bool]]],
        actions: Union[Callable[[], None], List[Callable[[], None]]]
    ) -> None:
        if not isinstance(conditions, list):
            conditions = [conditions]

        if not isinstance(actions, list):
            actions = [actions]

        self.rules.append({
            "conditions": conditions,  # type: ignore
            "actions": actions
        })

    def evaluate_rules(self) -> None:
        for rule in self.rules:
            conditions: List[Callable[[], bool]] = rule["conditions"]
            if all(condition() for condition in conditions):
                actions: List[Callable[[], None]] = rule["actions"]
                for action in actions:
                    action()
