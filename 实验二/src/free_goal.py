"""独立补测适配：不修改主实验的正权图约束。"""
from dataclasses import dataclass, asdict
from time import perf_counter
from types import MappingProxyType

from src.weighted_graph import WeightedGraph
from src.heuristic_tables import prepare
from src.core.astar import astar
from src.core.heuristics import manhattan

METHODS = ('corrected_manhattan', 'corrected_bfs', 'directed_landmark')
LABELS = ('修正曼哈顿', '修正BFS下界', '有向地标')


@dataclass(frozen=True)
class FreeGoalGraph:
    grid: tuple
    weights: object
    forward: object
    reverse: object
    minimum: int
    goal: tuple

    @classmethod
    def build(cls, grid, goal):
        # 先验证原始正权图，再仅把指定目标的进入代价改为0。
        positive = WeightedGraph.build(grid)
        goal = tuple(goal)
        if goal not in positive.weights:
            raise ValueError('终点必须可通行')
        weights = dict(positive.weights)
        weights[goal] = 0
        forward, reverse = {}, {p: [] for p in weights}
        for p in weights:
            edges = []
            for q, _ in positive.forward[p]:
                w = weights[q]
                edges.append((q, w))
                reverse[q].append((p, w))
            forward[p] = tuple(edges)
        raw = [list(row) for row in grid]
        raw[goal[0]][goal[1]] = 0
        minimum = min((v for p, v in weights.items() if p != goal), default=0)
        return cls(tuple(map(tuple, raw)), MappingProxyType(weights),
                   MappingProxyType(forward), MappingProxyType({p: tuple(v) for p,v in reverse.items()}),
                   minimum, goal)


def prepare_free(graph, method, landmarks):
    landmarks = tuple(map(tuple, landmarks))
    if method == METHODS[0]:
        return (lambda n: graph.minimum * max(manhattan(n, graph.goal)-1, 0)), 0, 0, 0
    if method == METHODS[1]:
        table = prepare(graph, graph.goal, 'bfs', landmarks)
        return (lambda n: graph.minimum * max(table.steps.get(n, 0)-1, 0)), table.entries, 1, 0
    if method == METHODS[2]:
        table = prepare(graph, graph.goal, 'landmark', landmarks)
        return table, table.entries, 0, table.dijkstra_calls
    raise ValueError('未知补测方法')


def session(graph, case, method, count, trace=None):
    # 每次独立调用冷启动方法专属表；公共图已统一构建。
    start = perf_counter()
    h, entries, bfs_calls, dijkstra_calls = prepare_free(graph, method, case['landmarks'])
    pre_end = perf_counter()
    metrics, queries, paths = {}, [], []
    for query in case['queries'][:count]:
        t = perf_counter()
        result = astar(tuple(query['start']), graph.goal, graph.forward.__getitem__,
                       lambda n,g: h(n), tie='fifo', trace=trace)
        seconds = perf_counter()-t
        m = asdict(result.metrics)
        queries.append({'query_id': query['id'], 'cost': result.cost, 'seconds': seconds, 'metrics': m})
        paths.append(result.path)
        for k,v in m.items():
            metrics[k] = max(metrics.get(k, 0), v) if k == 'queue_peak' else metrics.get(k, 0)+v
    end = perf_counter()
    return {'answer': [q['cost'] for q in queries], 'metrics': metrics, 'paths': paths, 'queries': queries,
            'preprocess_seconds': pre_end-start, 'search_seconds': end-pre_end, 'internal_seconds': end-start,
            'table_entries': entries, 'bfs_calls': bfs_calls, 'dijkstra_calls': dijkstra_calls, 'preparations': 1}
