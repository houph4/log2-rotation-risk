"""Rotation loss, 36-byte messages and query-independent sparse encoders."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize

SQRT2=np.sqrt(2.)
def svec(S):return np.r_[np.diag(S),SQRT2*S[0,1],SQRT2*S[0,2],SQRT2*S[1,2]]
def smat(v):
    S=np.diag(v[:3]);S[0,1]=S[1,0]=v[3]/SQRT2;S[0,2]=S[2,0]=v[4]/SQRT2;S[1,2]=S[2,1]=v[5]/SQRT2
    return S
def G(S):return np.trace(S)*np.eye(3)-S
def kconst(ell):
    c=np.cosh(ell/2)
    return c/1920+4/3*(c*c/1536+c/768)+c/(480*(1-ell*ell/16))
def polynomial(a,b,m,S):
    a,b=np.atleast_2d(a),np.atleast_2d(b)
    return np.sum(a*a-b*b,axis=1)-2*(a-b)@m-(np.einsum('ni,ij,nj->n',a,G(S),a)-np.einsum('ni,ij,nj->n',b,G(S),b))/12
def quat_losses(qa,qx):
    dot=np.abs(qa@qx.T)
    v=(qa[:,None,3,None]*qx[None,:,:3]-qx[None,:,3,None]*qa[:,None,:3]-np.cross(qa[:,None,:3],qx[None,:,:3]))
    return (2*np.arctan2(np.linalg.norm(v,axis=2),dot))**2
def point_losses(a,x):return quat_losses(Rot.from_rotvec(np.atleast_2d(a)).as_quat(),Rot.from_rotvec(x).as_quat())
def contrasts(a,b,x,w):return (point_losses(a,x)-point_losses(b,x))@w
def encode_moments(x,w):
    m=w@x;S=np.einsum('n,ni,nj->ij',w,x,x)
    return np.r_[m,np.diag(S),S[0,1],S[0,2],S[1,2]].astype('<f4').tobytes()
def decode_moments(wire):
    v=np.frombuffer(wire,dtype='<f4').astype(float)
    scaled=v[3:].copy();scaled[3:]*=SQRT2
    return v[:3],smat(scaled)
def matrix_state(x,w):return np.einsum('n,nij->ij',w,Rot.from_rotvec(x).as_matrix()).astype('<f4').tobytes()
def restore(wire):
    M=np.frombuffer(wire,dtype='<f4').astype(float).reshape(3,3)
    u,_,vh=np.linalg.svd(M);U=u@np.diag([1,1,np.linalg.det(u@vh)])@vh
    S=M+M.T+(1-np.trace(M))*np.eye(3)
    return (np.eye(3)-G(S)/12)@Rot.from_matrix(U).as_rotvec(),S
def indexed_wire(ids,weights):
    dtype=np.dtype([('id','<u2'),('weight','<f4')],align=False)
    z=np.zeros(6,dtype);z['id'][:len(ids)]=ids;z['weight'][:len(ids)]=weights
    wire=z.tobytes();assert len(wire)==36
    return wire
def decode_indexed(wire,x):
    z=np.frombuffer(wire,dtype=np.dtype([('id','<u2'),('weight','<f4')],align=False))
    w=z['weight'].astype(float);w/=w.sum()
    return x[z['id']],w
def medoids(x,w,k=6):
    D=point_losses(x,x);ids=[int(np.argmax(w))]
    while len(ids)<min(k,len(x)):
        score=w*D[:,ids].min(1);score[ids]=-1;ids.append(int(np.argmax(score)))
    ids=np.array(ids)
    for _ in range(20):
        labels=D[:,ids].argmin(1);new=ids.copy()
        for j in range(len(ids)):
            ii=np.flatnonzero(labels==j)
            if len(ii) and w[ii].sum()>0:new[j]=ii[np.argmin(D[np.ix_(ii,ii)]@w[ii])]
        if np.array_equal(new,ids):break
        ids=new
    labels=D[:,ids].argmin(1);mass=np.array([w[labels==j].sum() for j in range(len(ids))]);mass/=mass.sum()
    return indexed_wire(ids,mass)
def sparse_moments(x,w,k=6,r=np.deg2rad(29),return_diagnostics=False):
    phi=np.column_stack([x/r,np.array([svec(np.outer(v,v)) for v in x])/r**2]);target=w@phi
    ids=[];p=np.empty(0);approx=np.zeros(9);steps=[]
    for _ in range(min(k,len(x))):
        if not ids:j=int(np.argmin(np.sum((phi-target)**2,axis=1)))
        else:
            score=phi@(approx-target);score[ids]=np.inf;j=int(np.argmin(score))
        ids.append(j);A=phi[ids];p=np.r_[p,1.] if not len(p) else np.r_[p,0.]
        H=A@A.T;g=A@target
        res=minimize(lambda v:.5*v@H@v-g@v,p,jac=lambda v:H@v-g,
            constraints={'type':'eq','fun':lambda v:v.sum()-1,'jac':lambda v:np.ones(len(v))},
            bounds=[(0,1)]*len(ids),method='SLSQP',options={'maxiter':100,'ftol':1e-13})
        if not res.success and np.linalg.norm(res.x@A-target)>np.linalg.norm(p@A-target)+1e-9:raise RuntimeError(res.message)
        p=np.maximum(res.x,0);p/=p.sum();approx=p@A
        if return_diagnostics:
            steps.append({'support_size':len(ids),'selected_index':j,
                'success':bool(res.success),'status':int(res.status),'message':str(res.message),
                'iterations':int(res.nit),'objective':float(res.fun),
                'raw_min_weight':float(np.min(res.x)),
                'raw_weight_sum_error':float(abs(np.sum(res.x)-1)),
                'repaired_min_weight':float(np.min(p)),
                'repaired_weight_sum_error':float(abs(np.sum(p)-1)),
                'normalized_moment_error':float(np.linalg.norm(approx-target))})
    wire=indexed_wire(ids,p)
    if not return_diagnostics:return wire
    decoded_x,decoded_w=decode_indexed(wire,x)
    decoded_phi=np.column_stack([decoded_x/r,np.array([svec(np.outer(v,v)) for v in decoded_x])/r**2])
    return wire,{'residual_definition':'approximation-minus-target',
        'selected_indices':[int(v) for v in ids], 'steps':steps,
        'all_solver_steps_successful':all(s['success'] for s in steps),
        'pre_wire_normalized_moment_error':float(np.linalg.norm(approx-target)),
        'post_wire_normalized_moment_error':float(np.linalg.norm(decoded_w@decoded_phi-target)),
        'post_wire_min_weight':float(decoded_w.min()),
        'post_wire_weight_sum_error':float(abs(decoded_w.sum()-1))}

def wire_coordinate_errors(decoded):
    """Half the larger float32-neighbor gap, with differences computed in float64.

    Signed values, signed zero, subnormals and powers of two follow the same rule.
    At float32 range endpoints an infinite neighbor gives an infinite allowance.
    This covers nearest float32 transport; it is not an interval-arithmetic proof
    of the subsequent risk evaluation.
    """
    v=np.asarray(decoded,dtype=np.float32)
    if not np.all(np.isfinite(v)):raise ValueError('Wire entries must be finite')
    with np.errstate(over='ignore',under='ignore'):
        down=np.nextafter(v,np.float32(-np.inf),dtype=np.float32)
        up=np.nextafter(v,np.float32(np.inf),dtype=np.float32)
    vv=v.astype(np.float64)
    return .5*np.maximum(vv-down.astype(np.float64),up.astype(np.float64)-vv)

def wire_error_budgets(wire):
    v=np.frombuffer(wire,dtype='<f4')
    if len(v)!=9:raise ValueError('Log2 wire must contain nine float32 coordinates')
    err=wire_coordinate_errors(v)
    em=np.linalg.norm(err[:3])
    symmetric=err[3:].copy();symmetric[3:]*=SQRT2
    return float(em),float(np.linalg.norm(symmetric)),err
def build_messages(x,w):
    return {'Log mean':(w@x).astype('<f4').tobytes(),'Matrix9':matrix_state(x,w),
            'Log2':encode_moments(x,w),'Indexed-6 medoid':medoids(x,w),
            'Indexed-6 moment':sparse_moments(x,w),'FullPool':w.astype('<f4').tobytes()}
def prepare_message(method,wire,x,bank_quaternions=None):
    if method=='Log mean':return np.frombuffer(wire,dtype='<f4').astype(float),np.zeros((3,3))
    if method=='Matrix9':return restore(wire)
    if method=='Log2':return decode_moments(wire)
    if method.startswith('Indexed'):
        if bank_quaternions is None:
            xx,w=decode_indexed(wire,x);return Rot.from_rotvec(xx).as_quat(),w
        z=np.frombuffer(wire,dtype=np.dtype([('id','<u2'),('weight','<f4')],align=False));w=z['weight'].astype(float);w/=w.sum()
        return bank_quaternions[z['id']],w
    w=np.frombuffer(wire,dtype='<f4').astype(float);w/=w.sum()
    return (Rot.from_rotvec(x).as_quat() if bank_quaternions is None else bank_quaternions),w
def query_message(method,state,a,b):
    if method.startswith('Indexed') or method=='FullPool':
        qx,w=state;qa=Rot.from_rotvec(np.atleast_2d(a)).as_quat();qb=Rot.from_rotvec(np.atleast_2d(b)).as_quat()
        return (quat_losses(qa,qx)-quat_losses(qb,qx))@w
    return polynomial(a,b,*state)
def allowance(a,b,r,wire=None):
    a,b=np.atleast_2d(a),np.atleast_2d(b);d=a-b
    ell=r+np.maximum(np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1))
    E=2*kconst(ell)*np.linalg.norm(d,axis=1)*ell**5
    if wire is not None:
        em,eS,_=wire_error_budgets(wire)
        D=a[:,:,None]*a[:,None,:]-b[:,:,None]*b[:,None,:]
        GD=np.trace(D,axis1=1,axis2=2)[:,None,None]*np.eye(3)-D
        E+=2*np.linalg.norm(d,axis=1)*em+np.linalg.norm(GD,axis=(1,2))*eS/12
    return E
