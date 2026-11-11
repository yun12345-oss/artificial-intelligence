"""核对历史样本，并按报告口径重新计时。"""
import csv
import json
from collections import defaultdict
from math import ceil, isclose
from statistics import fmean, quantiles
from time import perf_counter

from experiments.common import ROOT

METRICS = (
    'expanded','visited','evaluated','prunes','astar_calls','path_expanded',
    'path_enqueued','path_queue_peak','distance_requests','order_requests',
    'transition_requests','bfs_preparations','dijkstra_preparations','heuristic_entries',
)
PHASES = ('elapsed','prepare_seconds','search_seconds','management_seconds','build_seconds')


def read_data(name):
    with (ROOT/'data/results'/name).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def counts(result):
    source = result.get('metrics',{}) | result
    return {key:source[key] for key in METRICS if key in source}


def compare_result(result, expected):
    label = expected['configuration']
    assert result['answer'] == json.loads(expected['answer']), (label,result['answer'])
    metrics = counts(result)
    for key in METRICS:
        if expected.get(key):
            assert metrics.get(key) == int(expected[key]), (label,key,metrics.get(key),expected[key])


def statistics(rows):
    output = {}
    for phase in PHASES:
        if rows[0].get(phase) in ('',None):
            continue
        samples = [float(row[phase]) for row in rows]
        lo,mid,hi = quantiles(samples,n=4,method='inclusive')
        output.update({
            phase+'_median':mid,phase+'_q1':lo,phase+'_q3':hi,phase+'_mean':fmean(samples),
        })
    return output


def verify_samples(summary):
    rows = read_data('timing_samples.csv')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['question'],row['configuration']].append(row)
    expected = {(r['question'],r['configuration']):r for r in summary}
    assert grouped.keys() == expected.keys()
    for key,group in grouped.items():
        record = expected[key]
        assert len(group)==9 and {int(r['batch']) for r in group}==set(range(9)), key
        for row in group:
            assert all(row.get(k,'')==record.get(k,'') for k in ('source_batch','answer',*METRICS)), key
        actual = statistics(group)
        for column,value in actual.items():
            assert isclose(value,float(record[column]),rel_tol=1e-12,abs_tol=1e-15), (key,column)
    return len(rows)


def benchmark(group, expected, batches, batch_size=None):
    probes = []
    for task in group:
        for _ in range(3):
            compare_result(task.execute(),expected[task.configuration])
        started = perf_counter()
        outcome = task.execute()
        probes.append(perf_counter()-started)
        compare_result(outcome,expected[task.configuration])
    if batch_size is None:
        batch_size = 1 if max(probes)>=.1 else min(10_000,max(1,ceil(.1/min(probes))))

    samples = {task.configuration:[] for task in group}
    for repeat in range(batches):
        shift = repeat%len(group)
        order = group[shift:]+group[:shift]
        if repeat%2:
            order.reverse()
        for task in order:
            started = perf_counter()
            results = [task.execute() for _ in range(batch_size)]
            elapsed = (perf_counter()-started)/batch_size
            for result in results:
                compare_result(result,expected[task.configuration])
            first = results[0]
            row = {'elapsed':elapsed}
            phase_names = (
                ('prepare_seconds',('prepare_seconds','preprocess_seconds','heuristic_prepare_seconds')),
                ('search_seconds',('search_seconds','game_seconds')),
                ('build_seconds',('build_seconds',)),
            )
            for dest,keys in phase_names:
                for key in keys:
                    if key in first:
                        row[dest] = fmean(result[key] for result in results)
                        break
            if task.configuration.split('/')[1] in ('corrected_manhattan','corrected_bfs','directed_landmark'):
                row['management_seconds'] = elapsed-fmean(result['internal_seconds'] for result in results)
            samples[task.configuration].append(row)
            del first, result, results

    return [{
        'configuration':task.configuration,
        'batches':batches,'batch_size':batch_size,
        **statistics(samples[task.configuration]),
    } for task in group]
