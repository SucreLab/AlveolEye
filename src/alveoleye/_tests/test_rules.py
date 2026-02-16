from alveoleye._rules import RulesEngine


def test_rules_engine_basic():
    engine = RulesEngine()

    flags = {"a": False, "b": False}

    def cond_true():
        return True

    def cond_false():
        return False

    def act_a():
        flags["a"] = True

    def act_b():
        flags["b"] = True

    # Only first rule should run (conditions all True)
    engine.add_rule([cond_true, cond_true], [act_a])
    engine.add_rule([cond_true, cond_false], [act_b])

    engine.evaluate_rules()

    assert flags["a"] is True
    assert flags["b"] is False


def test_rules_engine_coerce_single_condition_and_action():
    engine = RulesEngine()
    ran = {"x": False}

    def cond():
        return True

    def act():
        ran["x"] = True

    # Pass single callables, engine should coerce them to lists
    engine.add_rule(cond, act)
    engine.evaluate_rules()
    assert ran["x"] is True
