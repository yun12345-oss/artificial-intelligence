"""报告第三题的五种访问顺序。"""
from time import perf_counter

from src.core.alphabeta import alphabeta
from src.core.minimax import minimax
from src.core.tree_problem import TreeProblem

ORDERS = ('ideal', 'number', 'estimate', 'random', 'ideal_reverse')
SOLVERS = {'mm': minimax, 'ab': alphabeta}


class OrderedTree(TreeProblem):
    def __init__(self, base, order):
        self.nodes = base.nodes
        self.permutations = base.permutations
        self.ordering = {}
        self.keys = {}
        if order in ('ideal', 'estimate', 'ideal_reverse'):
            self.prepare('root', order)

    def prepare(self, path, order):
        node = self.nodes[path]
        if not node.children:
            value = node.value
            self.keys[path] = value
            return value, 1
        indices = list(range(len(node.children)))
        child_paths = [self.transition(path, i) for i in indices]
        leaf_totals = [self.prepare(child, order) for child in child_paths]
        total = sum(v for v, _ in leaf_totals)
        count = sum(n for _, n in leaf_totals)
        maximizing = self.is_max(path)
        values = [self.keys[child] for child in child_paths]
        self.keys[path] = total/count if order == 'estimate' else (max(values) if maximizing else min(values))
        sign = -1 if maximizing else 1
        if order == 'ideal_reverse':
            sign = -sign
        self.ordering[path] = sorted(indices, key=lambda i: (sign*values[i], i))
        return total, count

    def actions(self, state, order):
        if self.ordering:
            return self.ordering[state]
        return super().actions(state, order)


def run_task(base, depth, alg, order, trace=None):
    t0 = perf_counter()
    problem = OrderedTree(base, order)
    t1 = perf_counter()
    result = SOLVERS[alg](problem, 'root', depth, order, trace)
    t2 = perf_counter()
    return {
        'answer': {'value': result.value, 'action': result.action},
        'visited': result.metrics.visited, 'evaluated': result.metrics.evaluated,
        'prunes': result.metrics.prunes, 'prepare_seconds': t1-t0, 'search_seconds': t2-t1,
    }

