"""Aggregate existing per-target query caches without treating pairs as IID samples."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/competition'
f=pd.read_csv(OUT/'pair_metrics.csv');rows=[]
for (kind,method),g in f.groupby(['kind','method']):
    pairs=int(g.queries.sum());wrong=int(g.wrong.sum());penalty=float(g.regret_sum_deg2.sum())
    rows.append({'kind':kind,'method':method,'targets':g.sample_id.nunique(),
        'query_pairs':pairs,'wrong_pairs':wrong,'affected_targets':g.loc[g.wrong>0,'sample_id'].nunique(),
        'total_penalty_deg2':penalty,'mean_penalty_per_pair_deg2':penalty/pairs,
        'mean_penalty_per_error_deg2':penalty/wrong if wrong else 0.,
        'mean_target_penalty_deg2':float((g.regret_sum_deg2/g.queries).mean())})
summary=pd.DataFrame(rows);summary.to_csv(OUT/'target_error_summary.csv',index=False)
(OUT/'TARGET_ERROR_DEFINITION.json').write_text(json.dumps({
    'penalty':'indicator(readout and FullPool signs disagree) * abs(FullPool risk contrast)',
    'wrong_sign_tolerance_rad2':1e-12,'aggregation_unit':'distinct target; action pairs are nested',
    'per_pair_mean':'sum of penalties / number of pairs',
    'per_error_mean':'sum of penalties / wrong pairs',
    'target_mean':'mean over targets of (within-target sum of penalties / within-target pairs)'},indent=2))
print(summary[(summary.kind=='ICP all pairs')&summary.method.isin(['Log mean','Matrix9','Log2'])].to_string(index=False))
