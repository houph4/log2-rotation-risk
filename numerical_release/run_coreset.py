"""Same-bank, same-query36-byte moment and indexed-atom comparisons."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import json,time,argparse,hashlib
import numpy as np
import pandas as pd
from geometry import *
from run_competition import ordinary,boundary

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/coreset';OUT.mkdir(parents=True,exist_ok=True)
CONV=(180/np.pi)**2
def run(frontend,limit=None):
    paths=sorted((ROOT/'data/banks'/frontend).glob('*/*.npz'))
    expected=len(json.loads((ROOT/'BANK_INVENTORY.json').read_text())[frontend])
    if limit is None:assert len(paths)==expected,(frontend,len(paths),expected)
    results=[]
    for i,p in enumerate(paths[:limit]):
        rec=OUT/frontend/(p.stem+'.json');rec.parent.mkdir(parents=True,exist_ok=True)
        if rec.exists() and json.loads(rec.read_text()).get('wire_layout')=='unscaled-upper-v1':continue
        with np.load(p) as z:
            x=z['x'];w=z['w'];a=z['ordinary_a'] if 'ordinary_a' in z.files else ordinary(p.stem)[0]
            b=z['ordinary_b'] if 'ordinary_b' in z.files else ordinary(p.stem)[1]
            ca=z['boundary_a'] if 'boundary_a' in z.files else None;cb=z['boundary_b'] if 'boundary_b' in z.files else None
        if ca is None:ca,cb=boundary(p.stem,x,w)
        builders={'Log mean':lambda:(w@x).astype('<f4').tobytes(),'Matrix9':lambda:matrix_state(x,w),'Log2':lambda:encode_moments(x,w),
                  'Indexed-6 medoid':lambda:medoids(x,w),'Indexed-6 moment':lambda:sparse_moments(x,w),'FullPool':lambda:w.astype('<f4').tobytes()}
        messages={};cost={};states={}
        for name,f in builders.items():
            tick=time.perf_counter();messages[name]=f();cost[name]=(time.perf_counter()-tick)*1000
            tick=time.perf_counter();states[name]=prepare_message(name,messages[name],x);cost[name+'_decode']=(time.perf_counter()-tick)*1000
        assert len(messages['Log2'])==len(messages['Matrix9'])==len(messages['Indexed-6 moment'])==len(messages['Indexed-6 medoid'])==36
        qrows=[]
        for kind,qa,qb in [('ordinary',a,b),('boundary',ca,cb)]:
            if not len(qa):continue
            ref=contrasts(qa,qb,x,w)
            for name,state in states.items():
                val=query_message(name,state,qa,qb);err=val-ref;wrong=(np.sign(val)!=np.sign(ref))&(np.abs(ref)>1e-12)
                qrows.append({'frontend':frontend,'sample_id':p.stem,'dataset':p.parent.name,'kind':kind,'method':name,'queries':len(qa),
                    'sum_sq_error_rad4':float(err@err),'wrong':int(wrong.sum()),'binary_regret_sum_deg2':float(np.sum(np.abs(ref)*wrong)*CONV),
                    'payload':len(messages[name]),'build_ms':cost[name],'decode_ms':cost[name+'_decode']})
        # A common set of ordinary candidates is generated before risk evaluation.
        actions=np.vstack([a,b]);ref=point_losses(actions,x)@w;minrisk=ref.min();selection=[]
        for name,state in states.items():
            score=query_message(name,state,actions,np.zeros_like(actions));j=int(np.argmin(score))
            selection.append({'frontend':frontend,'sample_id':p.stem,'dataset':p.parent.name,'method':name,
                 'agree':bool(ref[j]-minrisk<=1e-12),'regret_deg2':float((ref[j]-minrisk)*CONV)})
        rec.write_text(json.dumps({'wire_layout':'unscaled-upper-v1','queries':qrows,'selection':selection},indent=2))
        if i%20==0:print('CORESET',frontend,i+1,len(paths),flush=True)
    collect(frontend)

def collect(frontend):
    qr=[];ss=[]
    for p in sorted((OUT/frontend).glob('*.json')):
        z=json.loads(p.read_text());qr+=z['queries'];ss+=z['selection']
    f=pd.DataFrame(qr);s=pd.DataFrame(ss)
    if len(f):f.to_csv(OUT/(frontend+'_queries.csv'),index=False);s.to_csv(OUT/(frontend+'_selection.csv'),index=False)
    print('CORESET_DONE',frontend,s.sample_id.nunique() if len(s) else 0,flush=True)

def timing(frontend):
    allpaths=sorted((ROOT/'data/banks'/frontend).glob('*/*.npz'));paths=[]
    for ds in ['lmo','icbin','tudl','tless','ycbv']:
        candidates=[p for p in allpaths if p.parent.name==ds]
        paths+=sorted(candidates,key=lambda p:hashlib.sha256(('timing82|'+p.stem).encode()).hexdigest())[:4]
    rows=[]
    for p in paths:
        with np.load(p) as z:x=z['x'];w=z['w']
        bank_quaternions=Rot.from_rotvec(x).as_quat()
        messages=build_messages(x,w);states={m:prepare_message(m,v,x,bank_quaternions) for m,v in messages.items()}
        builders={'Log mean':lambda:(w@x).astype('<f4').tobytes(),'Matrix9':lambda:matrix_state(x,w),'Log2':lambda:encode_moments(x,w),
                  'Indexed-6 medoid':lambda:medoids(x,w),'Indexed-6 moment':lambda:sparse_moments(x,w),'FullPool':lambda:w.astype('<f4').tobytes()}
        a,b=ordinary(p.stem,count=512)
        for method,state in states.items():
            bt=[];dt=[]
            for _ in range(2):
                warm_wire=builders[method]();prepare_message(method,warm_wire,x,bank_quaternions)
            for _ in range(7):
                start=time.perf_counter();wire=builders[method]();bt.append((time.perf_counter()-start)*1000)
                start=time.perf_counter();prepare_message(method,wire,x,bank_quaternions);dt.append((time.perf_counter()-start)*1000)
            for count in [1,8,32,128,512]:
                for _ in range(2):query_message(method,state,a[:count],b[:count])
                times=[]
                for _ in range(7):
                    start=time.perf_counter();value=query_message(method,state,a[:count],b[:count]);times.append((time.perf_counter()-start)*1000)
                rows.append({'frontend':frontend,'sample_id':p.stem,'dataset':p.parent.name,'method':method,'queries':count,
                             'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
                             'build_median_ms':float(np.median(bt)),'build_p95_ms':float(np.quantile(bt,.95)),
                             'decode_median_ms':float(np.median(dt)),'decode_p95_ms':float(np.quantile(dt,.95))})
    pd.DataFrame(rows).to_csv(OUT/(frontend+'_timing.csv'),index=False)
    print('TIMING_DONE',frontend,len(paths),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frontend',choices=['WAPR','FoundationPose'],default='WAPR');p.add_argument('--limit',type=int);p.add_argument('--timing',action='store_true');a=p.parse_args()
    timing(a.frontend) if a.timing else run(a.frontend,a.limit)
