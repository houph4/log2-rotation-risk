#!/usr/bin/env python3
"""Replay main Tables 2 and 3 from frozen aggregates; no model inference.

Portable archive layout (ROOT defaults to this script's directory):
  data/full_dense/WAPR_FIDELITY_SUMMARY.csv
  data/full_dense/FP_FULL_FIDELITY_SUMMARY.csv
  data/full_dense/dense_group_sensitivity.csv
  results/full_dense/table2.csv
  results/full_dense/table3.csv

Exact source-to-archive input map, relative to pose_evidence_theory:
  manuscript_visual_stage80/review/full_dense_deployment/
    full_dense_compact_20261027/fidelity_summary.csv
    -> data/full_dense/WAPR_FIDELITY_SUMMARY.csv
  manuscript_visual_stage80/evidence/full_dense_revision/fp_full/
    FP_FULL_FIDELITY_SUMMARY.csv
    -> data/full_dense/FP_FULL_FIDELITY_SUMMARY.csv
  revision_stage81/results/exposure/dense_group_sensitivity.csv
    -> data/full_dense/dense_group_sensitivity.csv

The FP summary combines the frozen FP_PARTITION_0_FIDELITY.csv and
FP_PARTITION_1_FIDELITY.csv in the same fp_full directory, as implemented by
manuscript_visual_stage80/scripts/update_fp_full_tables.py (lines 11-29).
WAPR full tables originate in scripts/integrate_full_dense_results.py.
The mixed equal_width_numbers.csv and boundary_numbers.csv retain an old
1,252-pool FP cohort and MUST NOT be substituted for these full FP inputs.

RMSE is aggregated as sqrt(sum(query_count * RMSE**2) / sum(query_count));
wrong-sign percentages are query weighted. RMSE units are degree squared,
not degrees. Group-excluded summaries already store squared error in radian
fourth powers, and are pooled before conversion to degree squared.

Table 2 contains the five datasets, pooled full population, and the pooled
group-excluded population. Table 3 contains the five reported full-population
readouts, followed by two explicitly marked group-excluded rows: only Matrix9
and Log2 are available in that saved exclusion summary. No unmeasured excluded
readouts or bootstrap intervals are inferred. Full-precision and printed-value
columns are both retained. The embedded publication-value check verifies the
rounded values of the current main tables, not a new experiment.
"""

import argparse
import csv
import json
import math
from pathlib import Path


DATASETS = (('lmo', 'LM-O'), ('icbin', 'IC-BIN'), ('tudl', 'TUD-L'),
            ('tless', 'T-LESS'), ('ycbv', 'YCB-V'))
READOUTS = (('Mean3', 'Log mean'),
            ('Matrix9_restore_float32', 'Restored Matrix9'),
            ('Log2_diagonal', 'Diagonal second moment'),
            ('Log2_isotropic', 'Isotropic second moment'),
            ('Log2_float32', 'Log2'))
RAD2_TO_DEG2 = (180.0 / math.pi) ** 2


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def full_rows(path, frontend):
    result = []
    for row in read_csv(path):
        n = int(row['queries'])
        rmse = (float(row['rmse_rad2']) * RAD2_TO_DEG2 if frontend == 'WAPR'
                else float(row['rmse_deg2']))
        wrong = (100.0 * (1.0 - float(row['sign_agree']))
                 if frontend == 'WAPR' else float(row['wrong_sign_percent']))
        result.append(dict(dataset=row['dataset'], query=row['query'],
                           method=row['method'], pools=int(row['valid_targets']),
                           queries=n, sse_deg4=n * rmse * rmse,
                           wrong_count=n * wrong / 100.0))
    return result


def aggregate(rows, query, method, dataset=None):
    selected = [r for r in rows if r['query'] == query and r['method'] == method
                and (dataset is None or r['dataset'] == dataset)]
    if not selected:
        raise ValueError(f'No summary rows for {dataset}, {query}, {method}')
    names = [r['dataset'] for r in selected]
    if len(names) != len(set(names)):
        raise ValueError(f'Duplicate dataset aggregates: {names}')
    n = sum(r['queries'] for r in selected)
    if n <= 0:
        raise ValueError('A query aggregate must have positive count')
    return dict(pools=sum(r['pools'] for r in selected), queries=n,
                rmse_deg2=math.sqrt(math.fsum(r['sse_deg4'] for r in selected) / n),
                wrong_percent=100.0 * math.fsum(r['wrong_count'] for r in selected) / n)


def excluded_rows(path, frontend):
    result = []
    mapping = {'Matrix9': 'Matrix9_restore_float32', 'Log2': 'Log2_float32'}
    for row in read_csv(path):
        if row['frontend'] != frontend or row['scope'] != 'exclude_dev_clusters':
            continue
        result.append(dict(dataset=row['dataset'],
                           query={'ordinary': 'regular', 'boundary': 'critical'}[row['query']],
                           method=mapping[row['method']], pools=int(row['targets']),
                           queries=int(row['queries']),
                           sse_deg4=float(row['sse']) * RAD2_TO_DEG2 ** 2,
                           wrong_count=float(row['wrong'])))
    return result


def make_table2(full, excluded):
    table = []
    for code, label in (*DATASETS, (None, 'Pooled'), (None, 'Group exclusion')):
        scope = 'exclude_dev_clusters' if label == 'Group exclusion' else 'all'
        sources = excluded if scope != 'all' else full
        out = dict(dataset=label, scope=scope)
        for frontend, prefix in (('WAPR', 'wapr'), ('FoundationPose', 'fp')):
            matrix = aggregate(sources[frontend], 'regular', 'Matrix9_restore_float32', code)
            log2 = aggregate(sources[frontend], 'regular', 'Log2_float32', code)
            if (matrix['pools'], matrix['queries']) != (log2['pools'], log2['queries']):
                raise ValueError('Matrix9 and Log2 populations differ')
            if matrix['queries'] != 128 * matrix['pools']:
                raise ValueError('Expected 128 ordinary queries per valid target')
            ratio = matrix['rmse_deg2'] / log2['rmse_deg2']
            out.update({f'{prefix}_pools': matrix['pools'],
                        f'{prefix}_ordinary_queries': matrix['queries'],
                        f'{prefix}_matrix9_rmse_deg2': matrix['rmse_deg2'],
                        f'{prefix}_log2_rmse_deg2': log2['rmse_deg2'],
                        f'{prefix}_ratio': ratio,
                        f'{prefix}_matrix9_printed': f"{matrix['rmse_deg2']:.4f}",
                        f'{prefix}_log2_printed': f"{log2['rmse_deg2']:.4f}",
                        f'{prefix}_ratio_printed': f'{ratio:.1f}'})
        table.append(out)
    return table


def make_table3(full, excluded):
    table = []
    for scope, sources, methods in (('all', full, READOUTS),
                                    ('exclude_dev_clusters', excluded,
                                     (READOUTS[1], READOUTS[-1]))):
        for method, label in methods:
            out = dict(readout=label, method=method, scope=scope)
            for frontend, prefix in (('WAPR', 'wapr'), ('FoundationPose', 'fp')):
                ordinary = aggregate(sources[frontend], 'regular', method)
                boundary = aggregate(sources[frontend], 'critical', method)
                base = aggregate(sources[frontend], 'critical', 'Matrix9_restore_float32')
                gain = base['wrong_percent'] - boundary['wrong_percent']
                out.update({f'{prefix}_pools': ordinary['pools'],
                            f'{prefix}_ordinary_queries': ordinary['queries'],
                            f'{prefix}_boundary_queries': boundary['queries'],
                            f'{prefix}_ordinary_wrong_percent': ordinary['wrong_percent'],
                            f'{prefix}_boundary_wrong_percent': boundary['wrong_percent'],
                            f'{prefix}_boundary_gain_pp': gain,
                            f'{prefix}_ordinary_printed': f"{ordinary['wrong_percent']:.4f}",
                            f'{prefix}_boundary_printed': f"{boundary['wrong_percent']:.2f}",
                            f'{prefix}_gain_printed': f'{gain:+.2f}'})
            table.append(out)
    return table


def check_published(table2, table3):
    expected2 = {
        'LM-O': (8985, '0.6125', '0.0083', '73.5', 8983, '0.5972', '0.0079', '76.0'),
        'IC-BIN': (2156, '1.1684', '0.0135', '86.7', 2156, '1.3150', '0.0143', '92.2'),
        'TUD-L': (23914, '0.1398', '0.0018', '79.7', 23912, '0.1406', '0.0020', '71.8'),
        'T-LESS': (66674, '0.8004', '0.0123', '65.0', 66674, '0.7291', '0.0118', '61.7'),
        'YCB-V': (98505, '0.5755', '0.0091', '63.4', 98505, '0.5527', '0.0092', '60.3'),
        'Pooled': (200234, '0.6404', '0.0098', '65.2', 200230, '0.6036', '0.0097', '62.5'),
        'Group exclusion': (173016, '0.6267', '0.0095', '65.7', 173013, '0.6055', '0.0095', '63.8'),
    }
    expected3 = {
        'Log mean': ('0.0573', '26.70', '+10.13', '0.0576', '27.99', '+6.84'),
        'Restored Matrix9': ('0.0275', '36.83', '+0.00', '0.0279', '34.83', '+0.00'),
        'Diagonal second moment': ('0.0068', '0.32', '+36.50', '0.0073', '0.24', '+34.59'),
        'Isotropic second moment': ('0.0523', '12.64', '+24.19', '0.0518', '12.63', '+22.20'),
        'Log2': ('0.0003', '0.00', '+36.83', '0.0004', '0.00', '+34.83'),
    }
    for row in table2:
        actual = tuple(row[f'{prefix}_{field}'] for prefix in ('wapr', 'fp')
                       for field in ('pools', 'matrix9_printed', 'log2_printed', 'ratio_printed'))
        if actual != expected2[row['dataset']]:
            raise ValueError(f"Table 2 publication mismatch: {row['dataset']}: {actual}")
    for row in table3:
        if row['scope'] != 'all':
            continue
        actual = tuple(row[f'{prefix}_{field}_printed'] for prefix in ('wapr', 'fp')
                       for field in ('ordinary', 'boundary', 'gain'))
        if actual != expected3[row['readout']]:
            raise ValueError(f"Table 3 publication mismatch: {row['readout']}: {actual}")


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--wapr-summary', type=Path, help='Override archived WAPR aggregate')
    parser.add_argument('--fp-summary', type=Path, help='Override archived FP aggregate')
    parser.add_argument('--group-summary', type=Path, help='Override archived exclusion aggregate')
    parser.add_argument('--check-only', action='store_true', help='Verify in memory; write no files')
    args = parser.parse_args()
    base = args.root / 'data/full_dense'
    paths = {'WAPR': args.wapr_summary or base / 'WAPR_FIDELITY_SUMMARY.csv',
             'FoundationPose': args.fp_summary or base / 'FP_FULL_FIDELITY_SUMMARY.csv'}
    group_path = args.group_summary or base / 'dense_group_sensitivity.csv'
    full = {name: full_rows(path, name) for name, path in paths.items()}
    excluded = {name: excluded_rows(group_path, name) for name in paths}
    table2, table3 = make_table2(full, excluded), make_table3(full, excluded)
    check_published(table2, table3)
    if not args.check_only:
        output = args.root / 'results/full_dense'
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / 'table2.csv', table2)
        write_csv(output / 'table3.csv', table3)
    print(json.dumps({'status': 'PASS', 'verification': 'all rounded main Table 2/3 values',
                      'table2_rows': len(table2), 'table3_rows': len(table3),
                      'full_valid_pools': {'WAPR': table2[5]['wapr_pools'],
                                           'FoundationPose': table2[5]['fp_pools']},
                      'mode': 'check-only' if args.check_only else 'csv-written'}, indent=2))


if __name__ == '__main__':
    main()
