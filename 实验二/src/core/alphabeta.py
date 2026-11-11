"""Fail-soft Alpha-beta；仅根结果视作精确决策，剪枝子树可能返回界。"""
import math
from .metrics import Metrics
from .minimax import GameResult
from .tracing import event


def alphabeta(problem, state, depth, order="number", trace=None):
    if type(depth) is not int or depth < 0:
        raise ValueError("深度必须为非负整数")
    metrics = Metrics()

    def visit(s, remaining, alpha, beta, path):
        metrics.visited += 1
        maximizing = problem.is_max(s)
        event(trace, "enter", path=path, maximizing=maximizing, depth=remaining, alpha=alpha, beta=beta)
        if remaining == 0 or problem.terminal(s):
            metrics.evaluated += 1
            value = problem.evaluate(s)
            event(trace, "evaluate", path=path, value=value)
            return value, None
        actions = problem.actions(s, order)
        best, chosen = (-math.inf if maximizing else math.inf), None
        cutoff = False
        for i, action in enumerate(actions):
            value, _ = visit(problem.transition(s, action), remaining - 1, alpha, beta, path + "/" + str(action))
            if (maximizing and value > best) or (not maximizing and value < best):
                best, chosen = value, action
            if maximizing:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            event(trace, "update", path=path, action=action, child_value=value, value=best, alpha=alpha, beta=beta)
            if alpha >= beta:
                cutoff = True
                if i + 1 < len(actions):
                    metrics.prunes += 1
                    event(trace, "prune", path=path, alpha=alpha, beta=beta, skipped=actions[i + 1:])
                break
        if hasattr(problem, "record"):
            problem.record(s, chosen, cutoff)
        return best, chosen

    value, action = visit(state, depth, -math.inf, math.inf, "root")
    return GameResult(value, action, metrics)
