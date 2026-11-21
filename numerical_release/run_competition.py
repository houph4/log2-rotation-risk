"""Generate fixed ICP actions first, then evaluate compact source-risk readers."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import sys,json,time,argparse,hashlib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.special import softmax
from scipy.optimize import brentq
from geometry import *

ROOT=Path(__file__).resolve().parent;V=ROOT.parent/'manuscript_visual_stage80'
OUT=ROOT/'results/competition';OUT.mkdir(parents=True,exist_ok=True)
P=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text());R=np.deg2rad(P['radius_deg']);CONV=(180/np.pi)**2
CONFIG=ROOT/'data/source_config.json'
TEMP=json.loads((CONFIG if CONFIG.exists() else ROOT.parent/'tpami_full_stage77/results/dense_development/TEMPERATURE_LOCK.json').read_text())['temperature']
def entries():
    p=ROOT/'data/competition/frames.json'
    return json.loads(p.read_text()) if p.exists() else json.loads((V/'evidence/cpu_completion/INPUT_INVENTORY.json').read_text())['frames']

def generate(limit=None,bop_root=None):
    from rgbd import mesh,crop,guard
    rows=entries()
    if bop_root:
        for row in rows:
            for key in ['rgb_path','depth_path','mask_path','mesh_path']:row[key]=str(Path(bop_root)/row[key])
    elif (ROOT/'data/competition/frames.json').exists():
        raise ValueError('Regenerating actions requires --bop-root pointing to downloaded BOP assets')
    sys.addaudithook(guard)
    for n,row in enumerate(rows[:limit]):
        sid=row['sample_id'];dest=OUT/(sid+'.npz')
        if dest.exists():continue
        t=time.perf_counter();v,_=mesh(row['mesh_path'],row['mesh_scale_to_m']);v=v[np.linspace(0,len(v)-1,min(P['model_vertices'],len(v))).astype(int)]
        dep,valid,K=crop(row);yy,xx=np.where(valid);z=dep[yy,xx];cloud=np.c_[(xx+.5-K[0,2])*z/K[0,0],(yy+.5-K[1,2])*z/K[1,1],z]
        tree=cKDTree(cloud);pose=np.asarray(row['initial_pose']);R0=Rot.from_matrix(pose[:3,:3]).as_matrix();t0=pose[:3,3]
        actions=[np.zeros(3)];names=['Initial'];translations=[t0];snapshots=[]
        def add(a,tt,name):
            if not any(np.linalg.norm(a-v)<1e-10 for v in actions):actions.append(a);translations.append(tt.copy());names.append(name)
        cached=ROOT/'data/competition/single_start'/(sid+'.npz')
        if not cached.exists():cached=V/'evidence/submission_trajectory'/(sid+'.npz')
        with np.load(cached) as old:
            for step in [1,2,5,10]:
                if step<len(old['aa']):add(old['aa'][step],old['translations'][step],f'cached_{step}')
        for start,offset in enumerate(P['initial_offsets_deg']):
            rr=R0@Rot.from_rotvec(np.deg2rad(offset)).as_matrix();tt=t0.copy();record=[np.zeros(3)]
            for step in range(1,P['icp_steps']+1):
                src=v@rr.T+tt;dist,idx=tree.query(src);keep=dist<=np.quantile(dist,P['trim_fraction']);a=src[keep];b=cloud[idx[keep]]
                if len(a)<4:break
                ac=a.mean(0);bc=b.mean(0);u,_,vh=np.linalg.svd((a-ac).T@(b-bc));dr=vh.T@np.diag([1,1,np.linalg.det(vh.T@u.T)])@u.T
                tt=dr@tt+bc-dr@ac;rr=dr@rr;aa=Rot.from_matrix(R0.T@rr).as_rotvec()
                if step in P['snapshot_steps']:add(aa,tt,f'start{start}_step{step}')
            for damping in P['damping']:add(aa*damping,t0+damping*(tt-t0),f'start{start}_damp{damping}')
        actions=np.asarray(actions);valid=np.linalg.norm(actions,axis=1)<=2*R
        np.savez_compressed(dest,actions=actions,names=np.asarray(names),translations=np.asarray(translations),eligible=valid,R0=R0)
        print('ACTIONS',n+1,len(rows),sid,len(actions),int(valid.sum()),round(time.perf_counter()-t,3),flush=True)

def ordinary(sid,count=128):
    seed=int.from_bytes(hashlib.sha256((sid+'|29|'+str(730701)).encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed);ab=rng.normal(size=(2,count,3));ab/=np.linalg.norm(ab,axis=2)[:,:,None];ab*=rng.random((2,count,1))**(1/3)*R
    return ab[0],ab[1]
def boundary(sid,x,w):
    seed=int.from_bytes(hashlib.sha256((sid+'|critical760204').encode()).digest()[:8],'little');rng=np.random.default_rng(seed)
    dirs=rng.normal(size=(16,3));dirs/=np.linalg.norm(dirs,axis=1)[:,None];a=[]
    for d in dirs:
        f=lambda t:float(contrasts(d*t,np.zeros(3),x,w)[0])
        if f(1e-5)>0:d=-d
        if f(1e-5)>=0 or f(R)<=0:continue
        root=brentq(f,1e-5,R,xtol=1e-13)
        for scale in [.999,1.001]:a.append(d*root*scale)
    a=np.array(a).reshape(-1,3);return a,np.zeros_like(a)

def evaluate(limit=None):
    records=[];pairs=[];bounds=[];rows=entries();allinputs=[]
    for n,row in enumerate(rows[:limit]):
        sid=row['sample_id']
        with np.load(OUT/(sid+'.npz')) as z:aa=z['actions'][z['eligible']];names=z['names'][z['eligible']];r0=z['R0']
        portable=ROOT/'data/competition/banks'/(sid+'.npz')
        if portable.exists():
            with np.load(portable) as z:x=z['x'];w=z['w']
        else:
            with np.load(V/'evidence/cpu_completion/raw'/(sid+'.npz')) as z:x=z['x'];w=softmax(-z['loss']/TEMP)
        messages=build_messages(x,w);states={m:prepare_message(m,z,x) for m,z in messages.items()}
        exact=point_losses(aa,x)@w;zero=np.zeros_like(aa)
        chosen={m:int(np.argmin(query_message(m,state,aa,zero))) for m,state in states.items()};full=int(np.argmin(exact));chosen['Reference']=0
        labels=ROOT/'data/competition/gt_labels.json'
        if labels.exists():xgt=np.asarray(json.loads(labels.read_text())[sid])
        else:
            gt=json.loads((Path(row['depth_path']).parent.parent/'scene_gt.json').read_text())[str(row['im_id'])][row['gt_index']]
            assert gt['obj_id']==row['obj_id'];rgt=np.asarray(gt['cam_R_m2c']).reshape(3,3);xgt=Rot.from_matrix(r0.T@rgt).as_rotvec()
        gtloss=point_losses(aa,xgt[None,:]).ravel()
        for method,j in chosen.items():
            records.append({'sample_id':sid,'dataset':row['dataset'],'cluster':row['cluster'],'method':method,'actions':len(aa),'selected':j,'selected_name':str(names[j]),
                'same_as_full':bool(j==full),'tied_with_full':bool(exact[j]-exact[full]<=1e-12),'regret_deg2':float((exact[j]-exact[full])*CONV),
                'source_gain_deg2':float((exact[0]-exact[j])*CONV),'GT_change_deg2':float((gtloss[j]-gtloss[0])*CONV)})
        A,B=ordinary(sid);C,D=boundary(sid,x,w);i,j=np.triu_indices(len(aa),1)
        queries={'ordinary':(A,B),'boundary':(C,D),'ICP all pairs':(aa[i],aa[j]),'ICP vs Initial':(aa[1:],np.zeros_like(aa[1:]))}
        for kind,(a,b) in queries.items():
            if not len(a):continue
            truth=contrasts(a,b,x,w);q=query_message('Log2',states['Log2'],a,b);E=allowance(a,b,R,messages['Log2'])
            resolved=np.abs(q)>E;err=np.abs(q-truth);assert np.max(err-E)<3e-13,(sid,kind,float(np.max(err-E)))
            assert not np.any(resolved&(np.sign(q)!=np.sign(truth)))
            for method,state in states.items():
                val=query_message(method,state,a,b);wrong=(np.sign(val)!=np.sign(truth))&(np.abs(truth)>1e-12)
                pairs.append({'sample_id':sid,'dataset':row['dataset'],'cluster':row['cluster'],'kind':kind,'method':method,'queries':len(a),'wrong':int(wrong.sum()),
                   'sum_sq_error_rad4':float(np.sum((val-truth)**2)),'regret_sum_deg2':float(np.sum(np.abs(truth)*wrong)*CONV)})
            for u in range(len(a)):
                bounds.append({'sample_id':sid,'dataset':row['dataset'],'cluster':row['cluster'],'kind':kind,'query':u,'q':q[u],'truth':truth[u],'E':E[u],'resolved':bool(resolved[u]),
                               'bound_accept':bool(q[u]+E[u]<0),'score_accept':bool(q[u]<0),
                               'GT_change_rad2':float(point_losses(a[u],xgt[None,:])[0,0]-point_losses(b[u],xgt[None,:])[0,0])})
        # Store query-independent source inputs for portable reproduction.
        allinputs.append({'sample_id':sid,'dataset':row['dataset'],'cluster':row['cluster']})
        print('COMPETITION',n+1,len(rows),sid,'actions',len(aa),'Log2/mean',chosen['Log2'],chosen['Log mean'],flush=True)
    pd.DataFrame(records).to_csv(OUT/'selection.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'pair_metrics.csv',index=False)
    pd.DataFrame(bounds).to_csv(OUT/'bounds.csv.gz',index=False,compression='gzip')
    f=pd.DataFrame(records);s=[]
    for (ds,m),g in f.groupby(['dataset','method']):
        s.append({'dataset':ds,'method':m,'targets':len(g),'agreement_percent':g.tied_with_full.mean()*100,'mean_regret_deg2':g.regret_deg2.mean(),
                  'p95_regret_deg2':g.regret_deg2.quantile(.95),'mean_source_gain_deg2':g.source_gain_deg2.mean(),'GT_change_deg2':g.GT_change_deg2.mean()})
    pd.DataFrame(s).to_csv(OUT/'selection_summary.csv',index=False)
    (OUT/'INPUTS.json').write_text(json.dumps(allinputs,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--bop-root',type=Path);p.add_argument('--action-dir',type=Path);a=p.parse_args()
    if a.action_dir:OUT=a.action_dir;OUT.mkdir(parents=True,exist_ok=True)
    generate(a.limit,a.bop_root) if a.generate else evaluate(a.limit)
