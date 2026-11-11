"""题目一：单位代价迷宫的三种启发函数。"""
from collections import deque
from dataclasses import asdict
from functools import partial
from time import perf_counter

from experiments.common import Task, json_value
from src.core.astar import astar
from src.core.heuristics import manhattan, zero
from src.heuristic_tables import prepare
from src.weighted_graph import WeightedGraph

METHODS = ('zero', 'manhattan', 'bfs')


def solve(graph, case, method, trace=None, open_trace=None):
    start = tuple(case['queries'][0]['start'])
    goal = tuple(case['goal'])
    started = perf_counter()
    if method == 'bfs':
        bound = prepare(graph, goal, 'bfs', case['landmarks'])

        def heuristic(node, _goal):
            return bound(node)
    else:
        heuristic = {'zero': zero, 'manhattan': manhattan}[method]
    prepared = perf_counter()
    result = astar(start, goal, graph.forward.__getitem__, heuristic,
                   trace=trace, open_trace=open_trace)
    searched = perf_counter()
    return {
        'answer': [result.cost],
        'path': result.path,
        'metrics': asdict(result.metrics),
        'preprocess_seconds': prepared-started,
        'search_seconds': searched-prepared,
        'internal_seconds': searched-started,
    }


def shortest_steps(grid, start, goal):
    """独立 BFS 直接读取原始地图。"""
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (row, col), steps = queue.popleft()
        if (row, col) == goal:
            return steps
        for r, c in ((row-1,col),(row,col+1),(row+1,col),(row,col-1)):
            if (0 <= r < len(grid) and 0 <= c < len(grid[0])
                    and grid[r][c] != 'X' and (r,c) not in seen):
                seen.add((r,c))
                queue.append(((r,c),steps+1))
    return -1


def groups(cases):
    for case_id, case in cases.items():
        graph = WeightedGraph.build(case['grid'])
        yield [Task(f'{case_id}/{method}/1',partial(solve,graph,case,method)) for method in METHODS]


def details(cases, reference):
    official = {'expansions':{}}
    for case in cases.values():
        goal = tuple(case['goal'])
        start = tuple(case['queries'][0]['start'])
        shortest = shortest_steps(case['grid'],start,goal)
        graph = WeightedGraph.build(case['grid'])
        for method in METHODS:
            trace, pending = [], []
            result = solve(graph,case,method,trace,pending)
            assert result['answer'] == [shortest]
            if case['id'] == 'maze_official':
                official['path'] = json_value(result['path'])
                assert official['path'] == reference['path']
                expanded = [e['node'] for e in trace if e['event']=='expand']
                official['expansions'][method] = json_value(expanded)
                assert official['expansions'][method] == reference['expansions'][method]
                if method == 'manhattan':
                    actual = {}
                    for i,node in enumerate(expanded):
                        key = f'{node[0]},{node[1]}'
                        if key in reference['key_open_queues']:
                            actual[key] = pending[i+1]
                    official['key_open_queues'] = json_value(actual)
                    assert official['key_open_queues'] == reference['key_open_queues']
    return official
