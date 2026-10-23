"""Frozen per-person extraction functions copied from verified V15/V9 sources."""
import numpy as np
from scipy.signal import welch,butter,sosfiltfilt
REGIONS={"frontal":[0,1,2,3,10,11],"central":[4,5],"temporal":[12,13,14,15],"parietal":[6,7],"occipital":[8,9]}
BANDS=[(1,4),(4,8),(8,13),(13,30),(30,45)]

PAIRS=[(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)]

def eeg_descriptors(eeg,power):
    x=np.asarray(eeg,float);n=len(x)
    absolute=power[:,:80].reshape(n,16,5);relative=power[:,80:].reshape(n,16,5)
    spectral=np.concatenate([np.c_[absolute[:,g].mean(1),relative[:,g].mean(1)] for g in REGIONS.values()],axis=1)
    asym=np.concatenate([absolute[:,a]-absolute[:,b] for a,b in PAIRS],axis=1)
    variance=np.var(x,axis=-1);d1=np.diff(x,axis=-1);d2=np.diff(d1,axis=-1)
    mobility=np.sqrt(np.var(d1,axis=-1)/np.maximum(variance,1e-12))
    complexity=np.sqrt(np.var(d2,axis=-1)/np.maximum(np.var(d1,axis=-1),1e-12))/np.maximum(mobility,1e-8)
    f,p=welch(x,fs=200,nperseg=400,axis=-1)
    keep=(f>=2)&(f<=40);prob=p[...,keep]/np.maximum(p[...,keep].sum(-1,keepdims=True),1e-20)
    entropy=-(prob*np.log(np.maximum(prob,1e-20))).sum(-1)/np.log(keep.sum())
    keep=(f>=2)&(f<=40)&~((f>=8)&(f<=13));lf=np.log(f[keep]);lf-=lf.mean()
    slope=np.sum(np.log(np.maximum(p[...,keep],1e-20))*lf,axis=-1)/np.sum(lf**2)
    down=x[...,::4];triples=np.stack([down[...,:-2],down[...,1:-1],down[...,2:]],axis=-1)
    order=np.argsort(triples,axis=-1,kind='stable');code=9*order[...,0]+3*order[...,1]+order[...,2]
    probs=np.stack([(code==k).mean(-1) for k in [5,7,11,15,19,21]],axis=-1)
    perm=-(probs*np.log(np.maximum(probs,1e-20))).sum(-1)/np.log(6)
    dynamics=np.stack([mobility,complexity,entropy,slope,perm],axis=-1)
    regional_dynamics=np.concatenate([dynamics[:,g].mean(1) for g in REGIONS.values()],axis=1)
    # Sensor-space absolute imaginary coherency, not source-localized connectivity.
    frames=np.lib.stride_tricks.sliding_window_view(x,200,axis=-1)[...,::100,:]
    fft=np.fft.rfft(frames*np.hanning(200),axis=-1);freq=np.fft.rfftfreq(200,1/200)
    coherence=[];region_values=list(REGIONS.values())
    for low,high in BANDS:
        z=fft[..., (freq>=low)&(freq<high)]
        cross=np.einsum('wcsk,wdsk->wcd',z,np.conjugate(z),optimize=True)
        diag=np.maximum(np.diagonal(cross.real,axis1=1,axis2=2),1e-20)
        im=np.abs(cross.imag)/np.sqrt(diag[:,:,None]*diag[:,None,:])
        for a,g in enumerate(region_values):
            for b in range(a,len(region_values)):
                h=region_values[b]
                ij=[(i,j) for i in g for j in h if a!=b or i<j]
                coherence.append(np.stack([im[:,i,j] for i,j in ij],axis=1).mean(1))
    connectivity=np.stack(coherence,axis=1)
    return np.c_[spectral,asym],regional_dynamics,connectivity,np.c_[spectral,asym,regional_dynamics]

EYE_NAMES=[f'{stat}_{axis}' for stat in ['sd','iqr','median_abs_speed','q90_abs_speed','mean_abs_acceleration','lag1_correlation'] for axis in ['x','y']]+[
    'radius_median','radius_q90','speed_median','speed_q90','straightness','speed_cv','direction_persistence',
    'geometry_anisotropy','low_frequency_fraction','linear_ramp_r2','msd_lag1','msd_lag8','msd_lag32']

def eye_descriptors(x):
    """Input normalized within measurement; all speeds are normalized-coordinate/s."""
    d=np.diff(x,axis=-1)*62.5;acc=np.diff(d,axis=-1)*62.5
    def corr(a,b):
        a=a-a.mean(-1,keepdims=True);b=b-b.mean(-1,keepdims=True)
        return np.sum(a*b,-1)/np.maximum(np.sqrt(np.sum(a*a,-1)*np.sum(b*b,-1)),1e-12)
    per_axis=np.concatenate([x.std(-1),np.quantile(x,.75,axis=-1)-np.quantile(x,.25,axis=-1),
        np.median(abs(d),axis=-1),np.quantile(abs(d),.9,axis=-1),np.mean(abs(acc),axis=-1),
        corr(x[...,:-1],x[...,1:])],axis=1)
    radius=np.sqrt(np.sum(x*x,axis=1));speed=np.sqrt(np.sum(d*d,axis=1))
    path=speed.sum(-1)/62.5;straight=np.sqrt(np.sum((x[:,:,-1]-x[:,:,0])**2,axis=1))/np.maximum(path,1e-8)
    direction=np.sum(d[:,:,1:]*d[:,:,:-1],axis=1)/np.maximum(speed[:,1:]*speed[:,:-1],1e-8)
    centered=x-x.mean(-1,keepdims=True);cov=centered@centered.transpose(0,2,1)/250
    eigen=np.linalg.eigvalsh(cov);anis=(eigen[:,1]-eigen[:,0])/np.maximum(eigen.sum(1),1e-8)
    spectrum=abs(np.fft.rfft(centered,axis=-1))**2;f=np.fft.rfftfreq(250,1/62.5)
    low=spectrum[...,(f>0)&(f<=2)].sum((1,2))/np.maximum(spectrum[...,f>0].sum((1,2)),1e-12)
    time=np.broadcast_to(np.arange(250),x.shape)
    ramp=(corr(x,time)**2).mean(1)
    msd=[np.mean(np.sum((x[:,:,lag:]-x[:,:,:-lag])**2,axis=1),axis=1) for lag in [1,8,32]]
    extra=np.c_[np.median(radius,1),np.quantile(radius,.9,axis=1),np.median(speed,1),np.quantile(speed,.9,axis=1),
        straight,speed.std(1)/np.maximum(speed.mean(1),1e-8),direction.mean(1),anis,low,ramp,*msd]
    return np.c_[per_axis,extra]

def eye_features(x,groups):
    all_rows=[];bag=[];split_a=[];split_b=[];multi=0
    for g in np.unique(groups):
        raw=np.asarray(x[groups==g,:2],float)
        # All windows in the SAME measurement share a center and isotropic scale.
        center=np.median(raw,axis=(0,2),keepdims=True)
        scale=max(float(np.sqrt(np.mean((raw-center)**2))),1e-6)
        rows=eye_descriptors((raw-center)/scale);all_rows.append((groups==g,rows))
        bag.append(np.r_[np.median(rows,axis=0),rows.std(0)])
        if len(rows)>=2:
            split_a.append(np.median(rows[::2],0));split_b.append(np.median(rows[1::2],0));multi+=1
    bag=np.stack(bag)
    window=np.empty((len(x),len(EYE_NAMES)),float)
    for mask,rows in all_rows:window[mask]=rows
    # Each measurement has equal weight; within-measurement SD is a separate block.
    d=len(EYE_NAMES)
    summary=np.r_[np.median(bag[:,:d],0),np.quantile(bag[:,:d],.75,axis=0)-np.quantile(bag[:,:d],.25,axis=0),
                  np.median(bag[:,d:],0)]
    sa=np.mean(split_a,0) if multi else np.zeros(d);sb=np.mean(split_b,0) if multi else np.zeros(d)
    return summary,bag,window,sa,sb,multi/max(len(bag),1)

def covariances(eeg):
    x=np.asarray(eeg,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(16,800) or len(x)<3 or not np.isfinite(x).all():raise ValueError('Invalid EEG')
    result=[]
    for low,high in BANDS:
        sos=butter(4,[low,high],btype='bandpass',fs=200,output='sos')
        # Filter each 4s epoch separately; no artificial concatenation of windows.
        # Default odd reflection padding, all 800 samples retained, no new ICA.
        z=sosfiltfilt(sos,x,axis=-1);z-=z.mean(-1,keepdims=True)
        c=z@z.transpose(0,2,1)/(z.shape[-1]-1)
        scale=np.trace(c,axis1=1,axis2=2)/16
        c=.9*c+.1*scale[:,None,None]*np.eye(16)
        result.append(c.mean(0)) # each PERSON is one observation, not each epoch
    out=np.stack(result)
    if np.linalg.eigvalsh(out).min()<=0:raise ValueError('Non-positive covariance')
    return out
