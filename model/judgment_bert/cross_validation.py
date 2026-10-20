"""Grouped K-fold inside development years, or expanding calendar-year validation."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from .balancing import balanced_bins, balance_report
from .analyze import iter_records
from .data import TASKS, labels_for, prepare, term_class


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def grouped_rows(path, task, first_year, test_year):
    rows = {}; parents = {}; keys = {}

    def find(line):
        while parents[line] != line:
            parents[line] = parents[parents[line]]
            line = parents[line]
        return line

    for line, row in iter_records(path):
        meta = row['meta']; year = meta.get('year'); fact = row['fact'].strip()
        if not isinstance(year, int) or not first_year <= year <= test_year or not fact:
            continue
        eligible = term_class(row) is not None if task == 'imprisonment' else bool(labels_for(row, task))
        if not eligible:
            continue
        rows[line] = row; parents[line] = line
        case = str(meta.get('case_number') or meta.get('source_url') or f'line:{line}')
        document = str(meta.get('source_url') or f'{case}:{meta.get("judgment_date", line)}')
        for key in [('case', case), ('document', document),
                    ('fact', hashlib.sha256(fact.encode()).hexdigest())]:
            if key in keys:
                parents[find(line)] = find(keys[key])
            else:
                keys[key] = line
    groups = defaultdict(list)
    for line in rows:
        groups[find(line)].append(line)
    return rows, list(groups.values())


def assignments(rows, groups, mode, folds, seed, test_year, task='accusation'):
    # Keep earlier development data; remove later holdout duplicates once, independently of fold.
    pool = []; holdout = []; removed = []
    for group in groups:
        development = [i for i in group if rows[i]['meta']['year'] < test_year]
        test = [i for i in group if rows[i]['meta']['year'] == test_year]
        if development:
            pool.append(development); removed.extend(test)
        else:
            holdout.extend(test)
    if not holdout:
        raise ValueError('No independent holdout records remain')
    base = {line: 'test' for line in holdout}
    plans = []
    if mode == 'kfold':
        bins = balanced_bins(rows, pool, folds, seed, task)
        for index, valid in enumerate(bins):
            split = dict(base)
            split.update((line, 'train') for group in pool for line in group)
            split.update((line, 'valid') for line in valid)
            plans.append((f'fold_{index+1:02d}', split))
    else:
        years = sorted({rows[i]['meta']['year'] for group in pool for i in group})
        for year in years[1:]:
            split = dict(base)
            for group in pool:
                train = [i for i in group if rows[i]['meta']['year'] < year]
                valid = [i for i in group if rows[i]['meta']['year'] == year]
                split.update((i, 'train') for i in train)
                # Purge validation records connected to earlier training records.
                if not train:
                    split.update((i, 'valid') for i in valid)
            if 'train' in split.values() and 'valid' in split.values():
                plans.append((f'valid_{year}', split))
        if not plans:
            raise ValueError('Need at least two development years for time validation')
    return plans, removed


def prepare_cv(data, task, output, mode='kfold', folds=5, seed=42, first_year=2018,
               test_year=2025, min_label_freq=30):
    if task not in TASKS or mode not in ('kfold', 'time'):
        raise ValueError('Unknown task or cross-validation mode')
    if min_label_freq < 1:
        raise ValueError('min-label-freq must be positive')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to replace CV directory: {output}')
    if first_year >= test_year:
        raise ValueError('first-year must precede test-year')
    rows, groups = grouped_rows(data, task, first_year, test_year)
    plans, removed = assignments(rows, groups, mode, folds, seed, test_year, task)
    output.mkdir(parents=True)
    info = dict(task=task, mode=mode, seed=seed, first_year=first_year, test_year=test_year,
                n_folds=len(plans), min_label_freq=min_label_freq,
                source_file=Path(data).name,
                source_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest(),
                grouping=['case_number', 'source_url/document', 'exact stripped fact SHA256; transitive union'],
                stratified=mode == 'kfold', removed_holdout_overlap_rows=len(removed), folds=[])
    if mode == 'kfold':
        info['balance'] = balance_report(rows, [[i for i,s in split.items() if s=='valid']
                                               for _,split in plans], task)
    for name, split in plans:
        manifest = prepare(data, task, output/name, test_year=test_year,
                           min_label_freq=min_label_freq, split_by_line=split)
        manifest['cross_validation'] = dict(mode=mode, fold=name, seed=seed,
                                            holdout_year=test_year, n_folds=len(plans))
        write_json(output/name/'manifest.json', manifest)
        write_json(output/name/'split_assignment.json', split)
        info['folds'].append(dict(name=name, num_labels=manifest['num_labels'],
                                 splits=manifest['splits']))
        print(f'{name}: labels={manifest["num_labels"]}, '+
              ', '.join(f'{s}={manifest["splits"][s]["n"]}' for s in ('train','valid','test')), flush=True)
    if mode == 'kfold':
        prepared_bins = [[json.loads(line)['source_line'] for line in
                          (output/f['name']/'valid.jsonl').read_text().splitlines()]
                         for f in info['folds']]
        info['balance_after_label_filtering'] = balance_report(rows, prepared_bins, task)
    write_json(output/'cv_manifest.json', info)
    return info


def summarize(prepared, output):
    plan = json.loads((Path(prepared)/'cv_manifest.json').read_text())
    results = []; missing = []; common_settings = None
    for fold in plan['folds']:
        name = fold['name']; path = Path(output)/name/'metrics.json'
        if not path.exists():
            missing.append(name); continue
        result = json.loads(path.read_text())
        if result.get('tiny_random_smoke') or result.get('fixed_epochs') or result.get('test_status') != 'not_evaluated':
            raise ValueError(f'{name}: smoke results or test evaluation cannot enter CV summary')
        trained_manifest = json.loads((Path(output)/name/'data_manifest.json').read_text())
        expected = json.loads((Path(prepared)/name/'manifest.json').read_text())
        if trained_manifest != expected:
            raise ValueError(f'{name}: training manifest does not match prepared fold')
        settings = json.loads((Path(output)/name/'run_config.json').read_text())
        comparable = {k:v for k,v in settings.items() if k not in ('data_dir','output','config')}
        if common_settings is not None and comparable != common_settings:
            raise ValueError('Cannot combine folds trained with different settings')
        common_settings = comparable
        best = next(epoch for epoch in result['history']
                    if epoch['valid']['macro_f1'] == result['best_valid_macro_f1'])
        results.append(dict(fold=name, selected_epoch=best['epoch'], valid=best['valid']))
    summary = dict(mode=plan['mode'], expected_folds=plan['n_folds'], completed_folds=len(results),
                   complete=not missing, missing_folds=missing, folds=results,
                   note='Unweighted mean and sample SD of selected validation scores; no TEST scores. '
                        'Fold label spaces are fitted independently; these are not pooled OOF metrics.')
    if results:
        summary['validation'] = {metric: dict(mean=statistics.mean(r['valid'][metric] for r in results),
            sample_std=statistics.stdev(r['valid'][metric] for r in results) if len(results)>1 else None)
            for metric in results[0]['valid'] if isinstance(results[0]['valid'][metric], (float, int))}
    Path(output).mkdir(parents=True, exist_ok=True)
    write_json(Path(output)/'cv_summary.json', summary)
    return summary


def train_cv(prepared, output, config, selected=None):
    prepared = Path(prepared).resolve(); output = Path(output).resolve()
    plan = json.loads((prepared/'cv_manifest.json').read_text())
    names = [f['name'] for f in plan['folds']]
    if selected is not None and selected not in names:
        raise ValueError(f'Unknown fold: {selected}')
    settings = json.loads(Path(config).read_text())
    if any(settings.get(key) for key in ('fixed_epochs','tiny_smoke','check_data')):
        raise ValueError('CV requires formal validation-based training, without fixed_epochs/tiny_smoke/check_data')
    settings['evaluate_test'] = False
    settings['expected_min_label_freq'] = plan['min_label_freq']
    run_plan = dict(settings={k:v for k,v in settings.items() if k not in ('data_dir','output')},
                    prepared_manifest=plan)
    output.mkdir(parents=True, exist_ok=True)
    provenance = output/'cv_run_manifest.json'
    if provenance.exists() and json.loads(provenance.read_text()) != run_plan:
        raise ValueError('Output already belongs to a different CV configuration or data plan')
    if not provenance.exists():
        write_json(provenance,run_plan)
    for name in names:
        if selected is not None and name != selected:
            continue
        run = output/name
        if run.exists():
            raise FileExistsError(f'Refusing to overwrite fold run: {run}')
        run.mkdir(parents=True)
        settings.update(data_dir=str(prepared/name), output=str(run))
        write_json(run/'cv_config.json', settings)
        subprocess.run([sys.executable, '-m', 'judgment_bert.train',
                        '--config', str(run/'cv_config.json'), '--no-evaluate-test'], check=True)
    return summarize(prepared, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--data', type=Path, required=True)
    prep.add_argument('--task', choices=TASKS, default='accusation')
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--mode', choices=['kfold', 'time'], default='kfold')
    prep.add_argument('--folds', type=int, default=5, help='K-fold count (default: 5; use 10 for ten-fold CV)')
    prep.add_argument('--seed', type=int, default=42)
    prep.add_argument('--first-year', type=int, default=2018)
    prep.add_argument('--test-year', type=int, default=2025)
    prep.add_argument('--min-label-freq', type=int, default=30)
    for command in ('train', 'summarize'):
        child = commands.add_parser(command)
        child.add_argument('--prepared', type=Path, required=True)
        child.add_argument('--output', type=Path, required=True)
        if command == 'train':
            child.add_argument('--config', type=Path, required=True)
            child.add_argument('--fold', help='Run one fold; omit to run all folds sequentially')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_cv(args.data, args.task, args.output, args.mode, args.folds, args.seed,
                   args.first_year, args.test_year, args.min_label_freq)
    elif args.command == 'train':
        print(json.dumps(train_cv(args.prepared, args.output, args.config, args.fold), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(summarize(args.prepared, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
