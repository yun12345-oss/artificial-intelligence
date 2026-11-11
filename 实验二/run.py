"""运行、核验与重新计时。所有命令从本目录读取数据。"""
import argparse
import json
import sys

sys.dont_write_bytecode = True

from experiments import q1, q2, q3, q4
from experiments.common import read_json
from experiments.measurement import benchmark, compare_result, counts, read_data, verify_samples

QUESTIONS = {'q1':q1,'q2':q2,'q3':q3,'q4':q4}


def inspect_question(question, references, historical):
    experiment = QUESTIONS[question]
    cases = read_json(f'data/inputs/{question}.json')
    expected = {r['configuration']:r for r in historical if r['question']==question}
    details = experiment.details(cases,references[question])
    if question=='q4':
        details['historical_selection'] = q4.historical_selection(expected.values())
    results = []
    checked = set()
    for group in experiment.groups(cases):
        for task in group:
            outcome = task.execute()
            if task.configuration in expected:
                compare_result(outcome,expected[task.configuration])
                checked.add(task.configuration)
            results.append({'configuration':task.configuration,'answer':outcome['answer'],**counts(outcome)})
    assert checked == expected.keys(), question
    return {'question':question,'details':details,'configurations':results,'historical_configurations':len(checked)}


def main():
    parser = argparse.ArgumentParser(description='复现报告四题的实验。')
    commands = parser.add_subparsers(dest='command',required=True)
    commands.add_parser('verify',help='核验四题结果并复算历史统计')
    run = commands.add_parser('run',help='输出指定题的确定性结果')
    run.add_argument('question',choices=QUESTIONS)
    timed = commands.add_parser('benchmark',help='为指定题重新计时')
    timed.add_argument('question',choices=QUESTIONS)
    timed.add_argument('--batches',type=int,default=9,help='正式批次，默认9')
    timed.add_argument('--batch-size',type=int,help='每批重复次数，默认按组校准')
    args = parser.parse_args(sys.argv[1:] or ["verify"])

    historical = read_data('report_summary.csv')
    references = read_json('data/results/reference.json')
    if args.command=='verify':
        sample_count = verify_samples(historical)
        for question in QUESTIONS:
            result = inspect_question(question,references,historical)
            print(f'{question}: {len(result["configurations"])} 个配置的确定性结果通过核验。')
        print(f'{len(historical)} 个历史配置、{sample_count} 条正式计时样本的统计核验通过。')
        return

    if args.command=='run':
        print(json.dumps(inspect_question(args.question,references,historical),ensure_ascii=False,indent=2))
        return

    if args.batches<2 or (args.batch_size is not None and args.batch_size<1):
        parser.error('批次数至少为2，每批重复次数至少为1。')
    question = args.question
    expected = {r['configuration']:r for r in historical if r['question']==question}
    fresh = []
    for group in QUESTIONS[question].groups(read_json(f'data/inputs/{question}.json')):
        group = [task for task in group if task.configuration in expected]
        if group:
            fresh.extend(benchmark(group,expected,args.batches,args.batch_size))
    print(json.dumps({'question':question,'time_unit':'seconds','new_measurements':fresh},ensure_ascii=False,indent=2))


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
