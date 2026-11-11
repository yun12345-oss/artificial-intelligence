"""与具体题目解耦的完整Minimax。适配器定义动作、转移与评价。"""
from dataclasses import dataclass
import math
from .metrics import Metrics
from .tracing import event


@dataclass
class GameResult:
    value: int
    action: object
    metrics: Metrics


def minimax(problem, state, depth, order="number", trace=None):
    if type(depth) is not int or depth < 0:
        raise ValueError("深度必须为非负整数")
    metrics = Metrics()

    def visit(s, remaining, path):
        metrics.visited += 1
        maximizing = problem.is_max(s)
        event(trace, "enter", path=path, maximizing=maximizing, depth=remaining)
        if remaining == 0 or problem.terminal(s):
            metrics.evaluated += 1
            value = problem.evaluate(s)
            event(trace, "evaluate", path=path, value=value)
            return value, None
        actions = problem.actions(s, order)
        best, chosen = (-math.inf if maximizing else math.inf), None
        for action in actions:
            value, _ = visit(problem.transition(s, action), remaining - 1, path + "/" + str(action))
            if (maximizing and value > best) or (not maximizing and value < best):
                best, chosen = value, action
            event(trace, "update", path=path, action=action, child_value=value, value=best)
        return best, chosen

    value, action = visit(state, depth, "root")
    return GameResult(value, action, metrics)
