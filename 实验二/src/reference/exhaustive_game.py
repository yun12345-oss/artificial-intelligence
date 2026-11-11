"""列举完整行动排列后按前缀回传；不调用正式状态转移或搜索。"""
from itertools import permutations


def evaluate(matrix,values,depth):
    length=min(depth,4)
    scores={}
    for seq in permutations(range(4),length):
        positions=[0,1]
        points=[0,0]
        for turn,a in enumerate(seq):
            player=turn%2
            points[player]+=values[a]-matrix[positions[player]][a+2]
            positions[player]=a+2
        scores[seq]=points[0]-points[1]
    levels={length:scores}
    for level in range(length-1,-1,-1):
        current={}
        for seq in permutations(range(4),level):
            children=[levels[level+1][seq+(a,)] for a in range(4) if a not in seq]
            current[seq]=(max if level%2 == 0 else min)(children)
        levels[level]=current
    value=levels[0][()]
    actions={a:levels[1][(a,)] for a in range(4)} if length else {}
    optimal=[a for a,v in actions.items() if v == value]
    prefix=()
    sequence=[]
    positions=[0,1]
    points=[0,0]
    for turn in range(length):
        a=next(a for a in range(4) if a not in prefix and levels[turn+1][prefix+(a,)] == levels[turn][prefix])
        player=turn%2
        cost=matrix[positions[player]][a+2]
        points[player]+=values[a]-cost
        sequence.append({"turn":turn+1,"player":player,"origin":positions[player],"target":a,
                         "cost":cost,"gain":values[a]-cost,"score_a":points[0],"score_b":points[1],"U":points[0]-points[1]})
        positions[player]=a+2
        prefix+=(a,)
    return {"value":value,"actions":actions,"optimal":optimal,"sequence":sequence}
