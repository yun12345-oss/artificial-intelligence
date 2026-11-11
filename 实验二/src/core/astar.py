"""A*核心与两道网格题适配。按进入格收费，允许更优g重新入队。"""
from dataclasses import dataclass
import heapq
import itertools
import math

from .heuristics import corrected, manhattan, zero
from .metrics import Metrics
from .tracing import event

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass
class PathResult:
    path: list
    cost: int
    metrics: Metrics


def validate(grid, start, goal):
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("地图必须是非空矩形")
    for p in (start, goal):
        if (not isinstance(p, (tuple, list)) or len(p) != 2
                or any(type(x) is not int for x in p)
                or not (0 <= p[0] < len(grid) and 0 <= p[1] < len(grid[0]))):
            raise ValueError("起终点必须是合法整数坐标")
        if grid[p[0]][p[1]] == "X":
            raise ValueError("起终点不可位于障碍")


def terrain_costs(grid, mode):
    if mode not in ("unit", "weighted", "treasure"):
        raise ValueError("未知地图模式")
    costs = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "X":
                continue
            if mode == "unit":
                if cell not in ("S", "E", ".", 1):
                    raise ValueError("单位代价地图包含未知格子")
                value = 1
            elif cell in ("S", "A", "B") or (isinstance(cell, str) and cell.startswith("T") and cell[1:].isdigit()):
                value = 1
            elif cell == "E":
                value = 0 if mode == "weighted" else 1
            else:
                try:
                    value = int(cell)
                except (ValueError, TypeError):
                    raise ValueError("未知地形") from None
                if isinstance(cell, bool) or (not isinstance(cell, str) and value != cell):
                    raise ValueError("仅支持非负整数权值")
            if value < 0:
                raise ValueError("不支持负权")
            if mode == "treasure" and value < 1:
                raise ValueError("综合题进入代价至少为1")
            costs[(r, c)] = value
    return costs


def astar(start, goal, neighbors, heuristic, tie="fifo", trace=None, *, open_trace=None):
    if tie not in ("fifo", "larger_g"):
        raise ValueError("未知平局策略")
    counter = itertools.count()
    scores, parents, heap = {start: 0}, {}, []
    metrics = Metrics()

    def push(node, g):
        serial = next(counter)
        key = (g + heuristic(node, goal), serial) if tie == "fifo" else (g + heuristic(node, goal), -g, serial)
        heapq.heappush(heap, (*key, node, g))
        metrics.enqueued += 1
        metrics.queue_peak = max(metrics.queue_peak, len(heap))

    push(start, 0)
    while heap:
        record = heapq.heappop(heap)
        node, g = record[-2:]
        if g != scores[node]:
            metrics.stale_pops += 1
            continue
        metrics.valid_pops += 1
        if open_trace is not None:
            # 记录逻辑优先队列。终点弹出前仍保留当时的所有有效旁支。
            pending = sorted(item for item in heap if scores[item[-2]] == item[-1])
            open_trace.append([
                (item[-2], item[-1], item[0] - item[-1], item[0])
                for item in (record, *pending)
            ])
        event(trace, "pop", node=node, g=g, h=heuristic(node, goal))
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parents[path[-1]])
            return PathResult(path[::-1], g, metrics)
        metrics.expanded += 1
        event(trace, "expand", node=node)
        for nxt, weight in neighbors(node):
            if weight < 0:
                raise ValueError("不支持负权")
            candidate = g + weight
            if candidate < scores.get(nxt, math.inf):
                scores[nxt], parents[nxt] = candidate, node
                push(nxt, candidate)
    return PathResult([], -1, metrics)


def search_grid(grid, start, goal, mode="unit", heuristic="informed", tie="fifo", trace=None):
    validate(grid, start, goal)
    start, goal = tuple(start), tuple(goal)
    costs = terrain_costs(grid, mode)
    if mode == "weighted":
        costs[goal] = 0
    if heuristic not in ("zero", "informed"):
        raise ValueError("未知启发函数")
    h = zero if heuristic == "zero" else manhattan
    if heuristic == "informed" and mode == "weighted":
        h = corrected(min((v for p, v in costs.items() if p != goal), default=0))

    def neighbors(p):
        for dr, dc in DIRECTIONS:
            nxt = (p[0] + dr, p[1] + dc)
            if nxt in costs:
                yield nxt, costs[nxt]

    return astar(start, goal, neighbors, h, tie, trace)


def maze_path(start, goal, grid):
    return search_grid(grid, start, goal).path


def weighted_cost(start, goal, grid):
    return search_grid(grid, start, goal, mode="weighted").cost
