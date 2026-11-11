"""题目四：O1 的查表与按需求路对照。"""
from dataclasses import asdict, dataclass, replace
from functools import partial
from hashlib import blake2b
from random import Random
from time import perf_counter

from experiments.common import Task, json_value
from src.core.alphabeta import alphabeta
from src.core.minimax import minimax
from src.core.treasure_game import State, TreasureGame
from src.heuristic_tables import prepare
from src.reference.dijkstra import matrix as reference_matrix
from src.reference.exhaustive_game import evaluate as exhaustive
from src.weighted_graph import WeightedGraph

HEURISTICS = ('manhattan', 'bfs', 'landmark')
SOLVERS = (('minimax','number'), ('alphabeta','number'),
           ('alphabeta','gain'), ('alphabeta','random'))
MODES = ('table', 'realtime')


@dataclass(frozen=True)
class Move:
    target: int
    gain: int | None = None

    def __str__(self):
        return str(self.target)


class RouteProvider:
    def __init__(self, graph, positions, heuristic, landmarks):
        self.graph = graph
        self.positions = positions
        self.method = heuristic
        self.landmarks = tuple(map(tuple,landmarks))
        self.by_goal = {}
        self.landmark_table = None
        self.table = None
        self.heuristic_prepare_seconds = 0.0
        self.stats = dict.fromkeys((
            'astar_calls','path_expanded','path_enqueued','path_queue_peak',
            'distance_requests','order_requests','transition_requests',
            'bfs_preparations','dijkstra_preparations','heuristic_entries',
        ),0)
        if heuristic == 'landmark':
            started = perf_counter()
            self.landmark_table = prepare(graph,positions[0],heuristic,self.landmarks)
            self.heuristic_prepare_seconds += perf_counter()-started
            self.stats['dijkstra_preparations'] = self.landmark_table.dijkstra_calls
            self.stats['heuristic_entries'] = self.landmark_table.entries

    def heuristic(self, goal):
        if self.landmark_table is not None:
            return replace(self.landmark_table,goal=goal)
        if self.method == 'manhattan':
            return prepare(self.graph,goal,self.method,self.landmarks)
        if goal not in self.by_goal:
            started = perf_counter()
            table = prepare(self.graph,goal,'bfs',self.landmarks)
            self.heuristic_prepare_seconds += perf_counter()-started
            self.by_goal[goal] = table
            self.stats['bfs_preparations'] += table.bfs_calls
            self.stats['heuristic_entries'] += table.entries
        return self.by_goal[goal]

    def search(self, source, target):
        goal = self.positions[target]
        result = self.graph.search(self.positions[source],goal,self.heuristic(goal))
        self.stats['astar_calls'] += 1
        self.stats['path_expanded'] += result.metrics.expanded
        self.stats['path_enqueued'] += result.metrics.enqueued
        self.stats['path_queue_peak'] = max(self.stats['path_queue_peak'],result.metrics.queue_peak)
        return result.cost

    def build_table(self):
        size = len(self.positions)
        self.table = [[0 if i==j else self.search(i,j) for j in range(size)] for i in range(size)]
        return self.table

    def distance(self, source, target, for_order=False):
        self.stats['distance_requests'] += 1
        self.stats['order_requests' if for_order else 'transition_requests'] += 1
        if self.table is not None:
            return self.table[source][target]
        return self.search(source,target)


class RoutedGame:
    """四个宝藏轮流领取，A 最大化得分差，B 最小化。"""
    def __init__(self, router, values, seed):
        self.router = router
        self.values = values
        self.seed = seed
        self.initial = State()

    def is_max(self, state):
        return state.player == 0

    def terminal(self, state):
        return state.remaining == 0

    def evaluate(self, state):
        return state.score_a-state.score_b

    def actions(self, state, order):
        candidates = [i for i in range(len(self.values)) if state.remaining & (1<<i)]
        if order == 'random':
            # 每个状态由固定种子决定顺序，不受其他分支的访问影响。
            key = blake2b(f'{self.seed};{state}'.encode()).digest()
            Random(int.from_bytes(key,'big')).shuffle(candidates)
        if order != 'gain':
            return [Move(i) for i in candidates]
        position = state.a if state.player==0 else state.b
        moves = [Move(i,self.values[i]-self.router.distance(position,i+2,for_order=True)) for i in candidates]
        return sorted(moves,key=lambda move:(-move.gain,move.target))

    def transition(self, state, move):
        player = state.player
        target = move.target
        position = state.a if player==0 else state.b
        gain = move.gain
        if gain is None:
            gain = self.values[target]-self.router.distance(position,target+2)
        common = {'remaining':state.remaining & ~(1<<target),'player':1-player}
        if player==0:
            return replace(state,a=target+2,score_a=state.score_a+gain,**common)
        return replace(state,b=target+2,score_b=state.score_b+gain,**common)


def solve(graph, case, depth, mode, heuristic, solver, order):
    started = perf_counter()
    positions = tuple(map(tuple,case['positions']))
    router = RouteProvider(graph,positions,heuristic,case['landmarks'])
    setup_done = perf_counter()
    if mode=='table':
        router.build_table()
    build_done = perf_counter()
    game = RoutedGame(router,case['values'],case['random_order_seed'])
    algorithm = minimax if solver=='minimax' else alphabeta
    result = algorithm(game,game.initial,depth,order)
    searched = perf_counter()
    return {
        'answer':{'value':result.value,'action':result.action.target},
        'metrics':asdict(result.metrics)|router.stats,
        'heuristic_prepare_seconds':router.heuristic_prepare_seconds,
        'build_seconds':build_done-setup_done if mode=='table' else 0.0,
        'game_seconds':searched-build_done,
        'internal_seconds':searched-started,
    }


def groups(case):
    graph = WeightedGraph.build(case['grid'])
    for depth in case['depths']:
        yield [
            Task(f'O1/{depth}/{mode}/{heuristic}/{solver}/{order}',
                 partial(solve,graph,case,depth,mode,heuristic,solver,order))
            for heuristic in HEURISTICS for solver,order in SOLVERS for mode in MODES
        ]


def details(case, reference):
    matrix = reference_matrix(case['grid'],case['positions'])
    assert matrix == reference['distance_matrix']
    graph = WeightedGraph.build(case['grid'])
    positions = tuple(map(tuple,case['positions']))
    for heuristic in HEURISTICS:
        router = RouteProvider(graph,positions,heuristic,case['landmarks'])
        assert router.build_table() == matrix
        assert router.stats['astar_calls']==30
    depths = {}
    game = TreasureGame(matrix,case['values'])
    for depth in case['depths']:
        answer = json_value(exhaustive(matrix,case['values'],depth))
        assert answer == reference['depths'][str(depth)]
        result = minimax(game,game.initial,depth)
        assert result.value == answer['value'] and result.action in answer['optimal']
        depths[str(depth)] = answer
    return {'distance_matrix':matrix,'depths':depths}

def historical_selection(records):
    """复算报告表5a的四项最小值和全部并列配置。"""
    criteria = {
        'elapsed_median':float,
        'astar_calls':int,
        'path_expanded':int,
        'visited':int,
    }
    winners = {}
    for depth in (2,4):
        group = [r for r in records if r['configuration'].split('/')[1]==str(depth)]
        winners[str(depth)] = {}
        for field,cast in criteria.items():
            minimum = min(cast(r[field]) for r in group)
            winners[str(depth)][field] = {
                'minimum':minimum,
                'configurations':[r['configuration'] for r in group if cast(r[field])==minimum],
            }
    return winners
