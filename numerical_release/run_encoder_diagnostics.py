"""Record optimizer termination, simplex feasibility and fitting error on all banks."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse
import json
import numpy as np
import pandas as pd
from geometry import sparse_moments,medoids,decode_indexed,svec

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/encoder_diagnostics';OUT.mkdir(parents=True,exist_ok=True)

def main(frontend):
    rows=[];step_rows=[];dest=OUT/frontend;dest.mkdir(exist_ok=True)
    paths=sorted((ROOT/'data/banks'/frontend).glob('*/*.npz'))
    for n,path in enumerate(paths):
        with np.load(path) as z:x=z['x'];w=z['w']
        r=np.deg2rad(29);phi=np.column_stack([x/r,np.array([svec(np.outer(v,v)) for v in x])/r**2]);target=w@phi
        wire,diag=sparse_moments(x,w,return_diagnostics=True)
        # Diagnostic instrumentation preserves the encoder's wire exactly.
        assert wire==sparse_moments(x,w)
        (dest/(path.stem+'.json')).write_text(json.dumps(diag,indent=2))
        base={'frontend':frontend,'sample_id':path.stem,'dataset':path.parent.name}
        for step in diag['steps']:step_rows.append({**base,**step})
        for method,message in [('Indexed-6 moment',wire),('Indexed-6 medoid',medoids(x,w))]:
            xx,ww=decode_indexed(message,x)
            received=np.column_stack([xx/r,np.array([svec(np.outer(v,v)) for v in xx])/r**2])
            row={**base,'method':method,'payload_bytes':len(message),'min_weight':float(ww.min()),
                 'weight_sum_error':float(abs(ww.sum()-1)),
                 'post_wire_normalized_moment_error':float(np.linalg.norm(ww@received-target))}
            if method.endswith('moment'):
                row.update(all_solver_steps_successful=diag['all_solver_steps_successful'],
                    pre_wire_normalized_moment_error=diag['pre_wire_normalized_moment_error'])
            rows.append(row)
        if n%100==0:print('ENCODER_DIAGNOSTICS',frontend,n+1,len(paths),flush=True)
    pd.DataFrame(rows).to_csv(OUT/(frontend+'_encodings.csv'),index=False)
    pd.DataFrame(step_rows).to_csv(OUT/(frontend+'_solver_steps.csv'),index=False)
    f=pd.DataFrame(rows);s=pd.DataFrame(step_rows)
    summary={'frontend':frontend,'targets':len(paths),'solver_steps':len(s),
        'success_steps':int(s.success.sum()),'failure_steps':int((~s.success).sum()),
        'statuses':{str(k):int(v) for k,v in s.status.value_counts().items()},
        'max_raw_weight_sum_error':float(s.raw_weight_sum_error.max()),
        'minimum_raw_solver_weight':float(s.raw_min_weight.min()),
        'max_received_weight_sum_error':float(f.weight_sum_error.max()),
        'minimum_received_weight':float(f.min_weight.min()),'methods':{}}
    for method,g in f.groupby('method'):
        e=g.post_wire_normalized_moment_error
        summary['methods'][method]={'mean_normalized_moment_error':float(e.mean()),
            'median_normalized_moment_error':float(e.median()),'p95_normalized_moment_error':float(e.quantile(.95)),
            'maximum_normalized_moment_error':float(e.max())}
    (OUT/(frontend+'_summary.json')).write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frontend',choices=['WAPR','FoundationPose'],required=True)
    main(p.parse_args().frontend)
