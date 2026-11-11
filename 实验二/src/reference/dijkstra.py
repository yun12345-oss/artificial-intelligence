"""直接读取原始网格，独立于被测图邻接、启发和Dijkstra实现。"""
import heapq
import math


def distances(grid,source,reverse=False):
    source=tuple(source)
    heap=[(0,source)]
    d={source:0}
    while heap:
        cost,(r,c)=heapq.heappop(heap)
        if cost != d[(r,c)]:
            continue
        for q in ((r-1,c),(r,c+1),(r+1,c),(r,c-1)):
            if not (0 <= q[0] < len(grid) and 0 <= q[1] < len(grid[0])) or grid[q[0]][q[1]] == "X":
                continue
            weight=grid[r][c] if reverse else grid[q[0]][q[1]]
            new=cost+weight
            if new < d.get(q,math.inf):
                d[q]=new
                heapq.heappush(heap,(new,q))
    return d


def matrix(grid,positions):
    return [[distances(grid,p)[tuple(q)] for q in positions] for p in positions]


def check_path(grid,path,start,goal,cost):
    assert path[0] == tuple(start) and path[-1] == tuple(goal)
    total=0
    for p,q in zip(path,path[1:]):
        assert abs(p[0]-q[0])+abs(p[1]-q[1]) == 1
        assert grid[q[0]][q[1]] != "X"
        total+=grid[q[0]][q[1]]
    assert total == cost
