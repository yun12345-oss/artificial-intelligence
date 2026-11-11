"""题目三：三棵生成树与官方小树的五顺序对照。"""
from dataclasses import asdict
from functools import partial

from experiments.common import Task
from src.core.minimax import minimax
from src.core.tree_problem import TreeProblem, from_dict
from src.tree_order import ORDERS, OrderedTree, run_task


def make_tree(case):
    problem = TreeProblem(from_dict(case['tree']),case['seed'] or 20261111)
    assert problem.permutations == case['permutations']
    return problem


def numbered_minimax(problem, depth):
    result = minimax(problem,problem.initial,depth)
    return {'answer':{'value':result.value,'action':result.action},'metrics':asdict(result.metrics)}


def groups(cases):
    for case_id,case in cases.items():
        tree = make_tree(case)
        depth = case['depth']
        tasks = [Task(f'{case_id}/ab/{order}',partial(run_task,tree,depth,'ab',order))
                 for order in ORDERS]
        if case_id == 'tree_official':
            yield tasks
            yield [Task(f'{case_id}/mm/number',partial(numbered_minimax,tree,depth))]
        else:
            tasks.append(Task(f'{case_id}/mm/number',partial(run_task,tree,depth,'mm','number')))
            yield tasks


def details(cases, reference):
    exact_roots = {}
    official = {}
    numbered_visits = {}
    for case_id,case in cases.items():
        tree = make_tree(case)
        depth = case['depth']
        baseline = minimax(tree,tree.initial,depth)
        assert baseline.value == reference['root_values'][case_id]
        exact_roots[case_id] = baseline.value
        if case_id != 'tree_official':
            trace = []
            run_task(tree,depth,'ab','number',trace)
            numbered_visits[case_id] = [e['path'] for e in trace if e['event']=='enter']
            continue
        node_values = OrderedTree(tree,'ideal').keys
        assert node_values == reference['official_node_values']
        for order in ORDERS:
            trace = []
            result = run_task(tree,depth,'ab',order,trace)
            assert result['answer']['value'] == baseline.value
            leaves = [e['value'] for e in trace if e['event']=='evaluate']
            cuts = [[e['path'],e['skipped']] for e in trace if e['event']=='prune']
            assert leaves == reference['official_leaves'][order]
            assert cuts == reference['official_prunes'][order]
            official[order] = {'leaf_values':leaves,'trace':trace}
    return {'root_values':exact_roots,'official_node_values':node_values,
            'official':official,'numbered_visited_paths':numbered_visits}
