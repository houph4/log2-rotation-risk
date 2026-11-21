"""Small-source and float32-neighbor checks for the transmitted interfaces."""
from pathlib import Path
from fractions import Fraction
import json
import numpy as np
from geometry import (sparse_moments, indexed_wire, decode_indexed, prepare_message,
                      query_message, contrasts, svec, encode_moments,
                      wire_coordinate_errors, wire_error_budgets, SQRT2)

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/implementation_checks';OUT.mkdir(parents=True,exist_ok=True)
checks={}
small_laws=[
    ('one_atom',np.array([[.1,-.05,.02]]),np.array([1.])),
    ('two_atoms',np.array([[.2,0,0],[-.2,0,0]]),np.array([.3,.7])),
    ('three_atoms',np.array([[.15,0,.02],[0,-.16,.01],[-.12,.08,-.03]]),np.array([.2,.3,.5])),
    ('four_atoms',np.array([[.1,.02,.03],[-.12,.04,.01],[.03,-.15,.06],[.02,.03,-.16]]),np.array([.1,.2,.3,.4])),
    ('coincident_atoms',np.repeat([[.1,-.05,.02]],4,axis=0),np.array([.1,.2,.3,.4])),
]
r=np.deg2rad(29)
rng=np.random.default_rng(830601)
a=rng.normal(0,.12,(16,3));b=rng.normal(0,.12,(16,3))
for name,x,w in small_laws:
    wire,diag=sparse_moments(x,w,return_diagnostics=True)
    assert len(wire)==36 and diag['all_solver_steps_successful']
    assert diag['post_wire_min_weight']>=0 and diag['post_wire_weight_sum_error']<1e-14
    assert diag['post_wire_normalized_moment_error']<2e-6,(name,diag)
    state=prepare_message('Indexed-6 moment',wire,x)
    error=float(np.max(np.abs(query_message('Indexed-6 moment',state,a,b)-contrasts(a,b,x,w))))
    assert error<1e-6,(name,error)
    # Independently check the known atom/weight wire and its risk readout.
    exact_wire=indexed_wire(np.arange(len(x)),w)
    xx,ww=decode_indexed(exact_wire,x)
    exact_error=float(np.max(np.abs(contrasts(a,b,xx,ww)-contrasts(a,b,x,w))))
    assert exact_error<1e-7
    checks[name]={'normalized_moment_error':diag['post_wire_normalized_moment_error'],
                 'maximum_query_error_rad2':error,'known_weight_wire_error_rad2':exact_error,
                 'solver_statuses':[s['status'] for s in diag['steps']]}

values=np.array([0.,-0.,np.nextafter(np.float32(0),np.float32(1)),
                 -np.nextafter(np.float32(0),np.float32(1)),
                 np.finfo(np.float32).tiny,-np.finfo(np.float32).tiny,
                 2.**-20,-2.**-20,.125,-.125,.5,-.5,1.,-1.],dtype=np.float32)
values=np.r_[values,np.nextafter(values,np.float32(np.inf)),
             np.nextafter(values,np.float32(-np.inf))].astype(np.float32)
eps=wire_coordinate_errors(values)
for v,e in zip(values,eps):
    down=np.nextafter(v,np.float32(-np.inf),dtype=np.float32)
    up=np.nextafter(v,np.float32(np.inf),dtype=np.float32)
    exact=max(Fraction.from_float(float(v))-Fraction.from_float(float(down)),
              Fraction.from_float(float(up))-Fraction.from_float(float(v)))/2
    assert Fraction.from_float(float(e))==exact,(v,e,exact)
checks['signed_zero_subnormal_power_neighbors']={'coordinates':len(values),
    'exact_rational_neighbor_budget_matches':True,'neighbor_dtype':'float32','difference_dtype':'float64'}
original=rng.uniform(-.7,.7,100_000)
received=original.astype(np.float32)
err=np.abs(received.astype(np.float64)-original)
budget=wire_coordinate_errors(received)
assert np.all(err<=budget)
checks['random_rounding']={'coordinates':len(original),'maximum_error_to_budget':float(np.max(err/budget))}
with np.errstate(over='ignore'):
    endpoint=wire_coordinate_errors(np.array([np.finfo(np.float32).max,-np.finfo(np.float32).max]))
assert np.isinf(endpoint).all()
checks['range_endpoints_have_conservative_infinite_budget']=True

x=small_laws[2][1];w=small_laws[2][2];wire=encode_moments(x,w)
em,eS,eps=wire_error_budgets(wire)
assert em==float(np.linalg.norm(eps[:3]))
assert np.isclose(eS**2,np.sum(eps[3:6]**2)+2*np.sum(eps[6:]**2),rtol=5e-16)
checks['symmetric_block_budget']={'epsilon_m':em,'epsilon_S':eS,'off_diagonal_factor':SQRT2}
(OUT/'UNIT_TESTS.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
