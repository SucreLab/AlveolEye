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
