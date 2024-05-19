from qtpy.QtWidgets import QLayout


class RulesEngine:
    def __init__(self):
        self.rules = []

    def add_rule(self, conditions, actions):
        if not isinstance(conditions, list):
            conditions = [conditions]

        if not isinstance(actions, list):
            actions = [actions]

        self.rules.append({"conditions": conditions, "actions": actions})

    def evaluate_rules(self):
        for rule in self.rules:
            conditions = rule["conditions"]
            if all(condition() for condition in conditions):
                actions = rule["actions"]
                for action in actions:
                    action()


def toggle(state, elements):
    if not isinstance(elements, list):
        elements = [elements]

    for item in elements:
        stack = [item]
        while stack:
            current = stack.pop()
            if isinstance(current, QLayout):
                for i in range(current.count()):
                    stack.append(current.itemAt(i).widget())
            else:
                current.setEnabled(state)
