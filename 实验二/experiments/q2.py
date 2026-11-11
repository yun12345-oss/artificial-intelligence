"""题目二：终点免费时的加权寻路与固定终点重复查询。"""
from functools import partial

from experiments.common import Task, json_value
from src.free_goal import FreeGoalGraph, METHODS, prepare_free, session
from src.reference.dijkstra import check_path, distances


def build_graph(case):
    # 冻结输入已写明终点免费。先恢复正权校验用的占位值，再由适配器设回0。
    goal = tuple(case['goal'])
    grid = [row[:] for row in case['grid']]
    row,col = goal
    assert grid[row][col] == 0
    grid[row][col] = 1
    graph = FreeGoalGraph.build(grid,goal)
    assert json_value(graph.grid) == case['grid']
    return graph


def groups(cases):
    for case_id,case in cases.items():
        graph = build_graph(case)
        counts = (1,) if case_id=='free_official' else (1,5,10,20)
        for count in counts:
            yield [Task(f'{case_id}/{method}/{count}',partial(session,graph,case,method,count))
                   for method in METHODS]


def details(cases, reference):
    official = {}
    for case_id,case in cases.items():
        graph = build_graph(case)
        truth = distances(graph.grid,graph.goal,reverse=True)
        count = 1 if case_id=='free_official' else 20
        for method in METHODS:
            estimate, *_ = prepare_free(graph,method,case['landmarks'])
            assert estimate(graph.goal) == 0
            assert all(0 <= estimate(p) <= cost for p,cost in truth.items())
            assert all(estimate(p) <= weight+estimate(q)
                       for p,edges in graph.forward.items() for q,weight in edges)
            trace = [] if case_id=='free_official' else None
            result = session(graph,case,method,count,trace=trace)
            for query,path,cost in zip(case['queries'][:count],result['paths'],result['answer'],strict=True):
                assert cost==truth[tuple(query['start'])]
                check_path(graph.grid,path,query['start'],graph.goal,cost)
            if case_id=='free_official':
                assert result['answer'] == [reference['cost']]
                assert json_value(result['paths'][0]) == reference['path']
                assert result['metrics']['expanded'] == reference['expansions'][method]
                for point in reference['heuristic_points']:
                    if point['method'] == method:
                        node = int(point['row']),int(point['col'])
                        assert truth[node] == int(point['true_cost'])
                        assert estimate(node) == int(point['estimate'])
                path = result['paths'][0]
                charges = [0]+[graph.weights[node] for node in path[1:]]
                official[method] = {
                    'path':path,'charges':charges,'cost':sum(charges),
                    'expanded':result['metrics']['expanded'],
                    'expansion_order':[e['node'] for e in trace if e['event']=='expand'],
                }
    return {'official':official,'heuristic_points':reference['heuristic_points']}
