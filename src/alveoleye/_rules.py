# alveoleye/_rules.py
from __future__ import annotations

from typing import Any, Callable, List, Sequence, Union
import alveoleye._gui_creator as gui_creator

Condition = Callable[[], bool]
Action = Callable[[], None]
ActionsLike = Union[Action, Sequence[Action]]

class RulesEngine:
    def __init__(self):
        self.rules: List[dict[str, Any]] = []
    
    def evaluate_rules(self) -> None:
        for rule in self.rules:
            conditions: Sequence[Condition] = rule["conditions"]
            if all(cond() for cond in conditions):
                actions: Sequence[Action] = rule["actions"]
                for action in actions:
                    action()

    def add_rule(self, conditions: Union[Condition, Sequence[Condition]], actions: ActionsLike) -> None:
        if not isinstance(conditions, (list, tuple)):
            conditions = [conditions]
        if not isinstance(actions, (list, tuple)):
            actions = [actions]
        self.rules.append({"conditions": conditions, "actions": actions})

    def toggle_visibility_based_on_condition(self, condition: Condition, widgets_or_layouts: Any) -> None:
        self.add_rule(condition, lambda: gui_creator.toggle(True,  widgets_or_layouts))
        self.add_rule(lambda: not condition(), lambda: gui_creator.toggle(False, widgets_or_layouts))

    def toggle_visibility_based_on_checkbox_state(self, checkbox: Any, widgets_or_layouts: Any) -> None:
        self.reevaluate_rules_when_qt_signals_emit(getattr(checkbox, "toggled", None))
        self.toggle_visibility_based_on_condition(lambda: bool(checkbox.isChecked()), widgets_or_layouts)

    def enable_or_disable_based_on_condition(self, condition: Condition, widgets: Any) -> None:
        enabler = getattr(gui_creator, "enable", None)
        
        if callable(enabler):
            self.add_rule(condition, lambda: enabler(True,  widgets))
            self.add_rule(lambda: not condition(), lambda: enabler(False, widgets))
        else:
            self.toggle_visibility_based_on_condition(condition, widgets)

    def enable_or_disable_based_on_checkbox_state(self, checkbox: Any, widgets: Any) -> None:
        self.reevaluate_rules_when_qt_signals_emit(getattr(checkbox, "toggled", None))
        self.enable_or_disable_based_on_condition(lambda: bool(checkbox.isChecked()), widgets)

    def reevaluate_rules_when_qt_signals_emit(self, *qt_signals: Any) -> None:
        for sig in qt_signals:
            try:
                if sig is not None:
                    sig.connect(self.evaluate_rules)
            except Exception:
                pass

    def run_actions_when_condition_is_true(self, condition: Condition, *actions: Action) -> None:
        if not actions:
            return
        
        self.add_rule(condition, list(actions))

    def toggle_visibility_when_any_condition_is_true(self, conditions: Sequence[Condition], targets: Any) -> None:
        self.toggle_visibility_based_on_condition(lambda: any(c() for c in conditions), targets)

    def toggle_visibility_when_all_conditions_are_true(self, conditions: Sequence[Condition], targets: Any) -> None:
        self.toggle_visibility_based_on_condition(lambda: all(c() for c in conditions), targets)

    def set_text_values_when_condition_is_true(self, condition: Condition, mapping_widget_to_text: dict[Any, str]) -> None:
        def _apply() -> None:
            for w, txt in mapping_widget_to_text.items():
                w.setText(txt)
       
        self.add_rule(condition, _apply)

    @staticmethod
    def condition_either(*conds: Condition) -> Condition:
        return lambda: any(c() for c in conds)

    @staticmethod
    def condition_all(*conds: Condition) -> Condition:
        return lambda: all(c() for c in conds)
