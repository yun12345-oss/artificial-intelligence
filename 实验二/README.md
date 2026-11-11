# 搜索工作量与总体成本：实验代码与数据

本项目对应五页报告《搜索工作量与总体成本：A*、Alpha-beta及宝藏博弈的实验分析》。四道题共用 A*、Minimax 和 Alpha-Beta 等算法。交付目录包含运行入口、固定地图与博弈树、独立校验代码、报告结果和正式计时样本。

## 1. 项目文件结构

```text
实验代码与数据/
├── README.md                        本说明
├── run.py                           命令行入口，支持 verify、run、benchmark
├── experiments/
│   ├── __init__.py
│   ├── common.py                    任务配置和目录内 JSON 读取
│   ├── measurement.py               批次计时、统计汇总和历史数据核对
│   ├── q1.py                        迷宫寻路、OPEN 队列和独立 BFS 校验
│   ├── q2.py                        免费终点加权寻路和重复查询
│   ├── q3.py                        博弈树回传与五种访问顺序对照
│   └── q4.py                        O1 的预建距离表与实时寻路组合
├── src/
│   ├── __init__.py
│   ├── weighted_graph.py            进入格收费的有向网格图
│   ├── heuristic_tables.py          反向 BFS 步数表和有向地标距离表
│   ├── free_goal.py                 免费终点计费、修正下界和查询复用
│   ├── tree_order.py                五种博弈树访问顺序及排序准备
│   ├── core/
│   │   ├── __init__.py
│   │   ├── astar.py                 A*、优先队列和路径回溯
│   │   ├── heuristics.py            零启发、曼哈顿和修正曼哈顿
│   │   ├── minimax.py               Minimax 数值回传
│   │   ├── alphabeta.py             Alpha-Beta 剪枝
│   │   ├── tree_problem.py          博弈树结构和固定随机排列
│   │   ├── treasure_game.py         宝藏博弈状态与按距离表的转移
│   │   ├── metrics.py               节点、叶子、剪枝等计数
│   │   └── tracing.py               运行时记录搜索事件
│   └── reference/
│       ├── __init__.py
│       ├── dijkstra.py              独立最短路和路径计费校验
│       └── exhaustive_game.py       穷举宝藏行动序列并回传精确值
└── data/
    ├── inputs/
    │   ├── q1.json                   官方迷宫、开阔图和绕行图
    │   ├── q2.json                   官方加权图和两张加权房间图
    │   ├── q3.json                   官方小树、生成树 A1–A3
    │   └── q4.json                   O1 地图、宝藏、深度与随机种子
    └── results/
        ├── reference.json           官方路径、关键队列、根值和精确决策
        ├── report_summary.csv       报告所用的 105 个历史配置统计
        └── timing_samples.csv       对应的 945 条正式计时样本
```

四道题从 `data/inputs/` 读取固定输入，由 `experiments/` 调用 `src/` 中的算法。核验时以独立方法和 `data/results/` 的参考记录检查结果。

## 2. 环境与运行命令

使用 Python 3.10 或更高版本，项目只依赖 Python 标准库。报告的原始计时环境是 Windows 11、Python 3.14.8。

在 Windows PowerShell 中运行：

```powershell
Set-Location -LiteralPath "$env:USERPROFILE\Desktop\实验代码与数据"
python --version
python run.py
```

直接执行 `python run.py` 等同于 `python run.py verify`。它会实际运行四题，核对答案和工作量，再从正式批次复算报告统计。需要查看某一题的路径、队列或博弈结果时，使用 `run` 子命令。

| 命令 | 输出 |
| --- | --- |
| `python run.py verify` | 四题核验摘要，包含确定性配置数和历史样本统计 |
| `python run.py run q1` | 返回路径、三种扩展顺序、曼哈顿在三个时点的 OPEN 队列，以及开阔与绕行图的结果 |
| `python run.py run q2` | 官方图每步入费、累计总代价、扩展顺序和启发值，以及房间图重复查询的结果 |
| `python run.py run q3` | 四棵树的根值、官方小树五种顺序的回传与剪枝事件，以及生成树按编号搜索的访问节点 |
| `python run.py run q4` | 6×6 距离矩阵、两种深度的精确首步评价及最优应对、48 种组合的工作量、报告表 5a 的历史选型 |
| `python run.py benchmark q4` | 第 4 题在本机的重新计时结果，换成 `q1`、`q2`、`q3` 可重测对应题目 |
| `python run.py -h` | 查看命令帮助。用 `python run.py benchmark -h` 查看计时参数 |

核验与按题运行会覆盖报告中的确定性实验。计时覆盖已保留历史计时的配置，题目一至四分别为 **6、27、24、48 组**。题目一的官方迷宫另有三种方法的路径与扩展校验，所以该题的核验输出为 9 组。

## 3. 本机重测与查看结果

重测第 4 题，将结果保存为桌面的 JSON 文件：

```powershell
python run.py benchmark q4 > "$env:USERPROFILE\Desktop\q4-本机重测.json"
```

重测四道题可以按顺序执行，每道题结束后生成一份 JSON：

```powershell
foreach ($question in 'q1', 'q2', 'q3', 'q4') {
    Write-Host "正在重测 $question"
    python run.py benchmark $question > "$env:USERPROFILE\Desktop\$question-本机重测.json"
    if ($LASTEXITCODE -ne 0) { break }
}
```

正式重测先对每个配置预热 3 次，再根据试运行耗时为同组配置校准统一的批大小，默认测 9 批。程序在每道题全部结束后输出 JSON。只想快速检查计时入口时，可用 `python run.py benchmark q4 --batches 2 --batch-size 1`，其中 `--batches` 指正式批次数，`--batch-size` 指每批重复执行次数。

重测 JSON 的 `new_measurements` 是各配置的统计结果。下面的命令把第 4 题的总耗时中位数换算为毫秒，并按耗时排序：

```powershell
$measurement = Get-Content -Raw "$env:USERPROFILE\Desktop\q4-本机重测.json" | ConvertFrom-Json
$measurement.new_measurements |
    Sort-Object elapsed_median |
    Select-Object configuration, batches, batch_size,
        @{Name='总耗时中位数 (ms)'; Expression={[math]::Round($_.elapsed_median * 1000, 3)}} |
    Format-Table -AutoSize
```

比对同一配置时，以 `configuration` 连接重测结果与 `data/results/report_summary.csv`。历史统计可以用原样本精确复算。本机的时间数值随机器和负载变化，答案、路径、访问量与 A* 调用次数用来核对实验是否执行一致。

## 4. 报告与输入、结果的对应关系

| 报告页 | 输入和实验范围 | 可直接核对的结果 |
| --- | --- | --- |
| 第 1 页，迷宫寻路 | 官方迷宫及种子 20261111 的两张 60×60 图。大图均使用首个查询 | 官方路径 7 步，零启发、曼哈顿、BFS 下界分别扩展 15、13、7 次。绕行图 43 步，分别扩展 1,035、509、278 次 |
| 第 2 页，加权地形 | 官方图及同种子的 30×30、60×60 房间图。目标格免费，房间图使用固定查询顺序的前 1、5、10、20 项 | 官方总代价 7，三种下界分别扩展 10、10、8 次。60×60 图前 20 次查询分别扩展 10,378、8,559、4,716 次 |
| 第 3 页，博弈剪枝 | 官方小树及 A1–A3。对照理想顺序、原始编号、估值顺序、固定随机顺序、理想逆序和编号 Minimax 基线 | A1–A3 根值为 −8、−5、−11。官方根值为 5，五种 Alpha-Beta 顺序分别访问 11、14、11、12、15 个节点 |
| 第 4–5 页，宝藏博弈 | O1 的深度 2 和 4，分别比较两种距离模式、三种启发函数与四种博弈策略 | 两种深度的最优得分差为 −1 和 3，首步分别取 T2 和 T3。共 48 个组合 |

第三题的官方编号 Minimax 基线来自独立批次，历史总耗时中位数为 37.08 μs。第四题的 O1 运行入口依据现有算法和固定随机种子 3101 补写，其 48 个配置的最优动作、访问与叶子计数、剪枝计数以及 A* 调用与扩展量均已对照历史数据。

## 5. 读懂数据字段

三份结果文件各有用途。需要复核路径和精确决策时看 `reference.json`，查报告数值时看 `report_summary.csv`，复算统计时看 `timing_samples.csv`。

| 字段 | 含义 |
| --- | --- |
| `question`、`configuration` | 题号和配置标识。配置名的格式见下方 |
| `source_batch` | 原正式数据的批次来源 |
| `batch`、`batch_size`、`execution_position` | 批次编号、该批重复次数、该配置在批内的执行位置。编号从 0 开始 |
| `answer` | 寻路为各查询的最优步数或路费，博弈为最优值与选择的动作 |
| `expanded` | A* 扩展的有效节点数。终点出队时直接回溯路径，计数停在此前 |
| `visited`、`evaluated`、`prunes` | 博弈访问节点数（含根和叶）、实际评价的叶数、跳过后续子节点的剪枝次数 |
| `astar_calls`、`path_expanded` | 综合题实际调用 A* 的次数、这些调用的累计扩展数 |
| `path_enqueued`、`path_queue_peak` | 综合题所有 A* 调用的累计入队数、各次 A* 运行中的最大队列峰值 |
| `distance_requests`、`order_requests`、`transition_requests` | 请求路费的总次数、排序时的请求次数、状态转移时的请求次数。预建模式的请求可直接查表 |
| `bfs_preparations`、`dijkstra_preparations`、`heuristic_entries` | 综合题构建 BFS 表的次数、地标正反 Dijkstra 次数、启发函数表的条目数 |
| `elapsed` | 每批壁钟耗时除以批大小，单位为秒 |
| `prepare_seconds`、`search_seconds`、`build_seconds`、`management_seconds` | 对应题目记录的准备、搜索、距离表构建和管理耗时，单位为秒 |

同一字段后缀 `_median`、`_q1`、`_q3` 和 `_mean` 分别表示中位数、第一四分位数、第三四分位数和均值。四分位采用 Python `statistics.quantiles(..., method='inclusive')`。秒乘以 1,000 得毫秒（ms），乘以 1,000,000 得微秒（μs）。空白表示源数据没有记录该项。

| 题目 | `configuration` 格式 | 例子 |
| --- | --- | --- |
| q1、q2 | 案例/启发函数/查询数 | `H2_60_20261111/manhattan/1` |
| q3 | 树/求解器/访问顺序 | `tree_official/ab/ideal` |
| q4 | O1/深度/距离模式/启发函数/求解器/访问顺序 | `O1/4/table/manhattan/alphabeta/number` |

坐标按“行、列”表示，从 0 开始。第四题的动作 0 至 3 对应 T1 至 T4，`table` 表示预建距离表，`realtime` 表示按需寻路，`gain` 表示按宝藏价值减路费的净收益从高到低排列。第三题的 `estimate` 用子树叶值均值排序，`ideal` 与 `ideal_reverse` 使用精确子树值。

## 6. 计时边界与历史数据来源

- **题目一**计入启发函数准备与 A* 搜索。历史样本取自原项目 `progressive/results/raw` 的两张 60×60 图单查询批次。
- **题目二**在计时前构建各策略共享的免费终点图。每次任务重新准备方法专属表，同一任务内的固定终点查询复用该表。历史样本来自“免费终点三启发函数补测”。
- **题目三**生成树和官方五顺序的总时间计入排序准备。历史样本来自报告使用的两次五顺序补测，官方编号 Minimax 来自 `group_044`。
- **题目四**总时间覆盖启发函数准备、实际寻路或预建表，以及博弈搜索。预建表每组先求 30 个非对角有向距离，实时模式在请求路费时调用 A*。按需构建 BFS 下界的时间可能落在建表或博弈阶段，比较完整任务时读取 `elapsed_median`。历史样本来自“题目四_O1同轮测量”。
