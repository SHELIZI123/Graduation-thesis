"""PPG shape and native-unit / dimensionless interval features, never BPM or ms HRV."""
import numpy as np

SHAPE_NAMES = ['q05','q25','q50','q75','q95','mean','sd','iqr','mad','skew','excess_kurtosis',
               'median_abs_diff','rms_diff','q95_abs_diff','rms_second_diff','crossing_rate',
               'ac1','ac4','ac16','ac32','spectral_entropy','spectral_centroid_cycles_per_sample',
               'dominant_cycles_per_sample','low_spectral_fraction','high_spectral_fraction']
NNI_NAMES = ['native_median','native_iqr','relative_mean','relative_sd','relative_q05','relative_q25',
             'relative_q75','relative_q95','relative_mad','relative_diff_rms','relative_diff_abs_mean',
             'relative_diff_q95','ac1','ac2','ac4','poincare_relative_short','poincare_relative_long',
             'relative_skew','relative_excess_kurtosis','relative_hist_entropy','fraction_rejected']
PPG_DIM = 2*len(SHAPE_NAMES)
HEART_NAMES = ['ppg_'+p+'_'+n for p in ['median','sd'] for n in SHAPE_NAMES] + ['nni_'+n for n in NNI_NAMES] + ['is_ppg','is_nni']
HEART_DIM = len(HEART_NAMES)

def ac(x, lag):
    if len(x) <= lag+1: return 0.
    a,b = x[:-lag]-x[:-lag].mean(),x[lag:]-x[lag:].mean()
    return float(np.sum(a*b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))

def moments(x):
    z = (x-x.mean())/max(float(x.std()),1e-8)
    return float(np.mean(z**3)), float(np.mean(z**4)-3)

def shape(x):
    x = np.asarray(x,float)
    # Interpolation is measured in samples. No time-axis unit is assumed.
    valid=np.isfinite(x)
    if valid.mean()<.98 or valid.sum()<64: return None
    bad=np.where(~valid)[0]
    if len(bad):
        runs=np.split(bad,np.where(np.diff(bad)>1)[0]+1)
        if max(map(len,runs))>2 or not valid[0] or not valid[-1]: return None
        x=np.interp(np.arange(len(x)),np.where(valid)[0],x[valid])
    med=np.median(x); scale=np.quantile(x,.75)-np.quantile(x,.25)
    if scale<1e-8: return None
    x=np.clip((x-med)/scale,-8,8); diff=np.diff(x); second=np.diff(diff)
    power=abs(np.fft.rfft((x-x.mean())*np.hanning(len(x))))**2
    freq=np.fft.rfftfreq(len(x)); power[0]=0; p=power/max(float(power.sum()),1e-12)
    entropy=-np.sum(p*np.log(np.maximum(p,1e-12)))/np.log(len(p))
    skew,kurt=moments(x)
    return np.r_[np.quantile(x,[.05,.25,.5,.75,.95]),x.mean(),x.std(),np.quantile(x,.75)-np.quantile(x,.25),
                 np.median(abs(x-np.median(x))),skew,kurt,np.median(abs(diff)),np.sqrt(np.mean(diff**2)),
                 np.quantile(abs(diff),.95),np.sqrt(np.mean(second**2)),np.mean(x[1:]*x[:-1]<0),
                 *[ac(x,l) for l in [1,4,16,32]],entropy,np.sum(p*freq),freq[np.argmax(p)],
                 p[(freq>0)&(freq<=.05)].sum(),p[freq>.2].sum()]

def ppg_record(raw):
    if len(raw)<256: return None
    starts=np.unique(np.linspace(0,len(raw)-256,min(32,max(1,len(raw)//256))).astype(int))
    features=[v for s in starts if (v:=shape(raw[s:s+256])) is not None]
    if not features: return None
    x=np.stack(features)
    return np.r_[np.median(x,0),x.std(0)]

def nni_record(raw):
    raw=np.asarray(raw,float); positive=np.isfinite(raw)&(raw>0)
    if positive.sum()<12: return None
    med=np.median(raw[positive]); valid=positive&(raw>=.25*med)&(raw<=4*med)
    # Never concatenate across rejected intervals for successive-difference features.
    chunks=[v for v in np.split(np.where(valid)[0],np.where(np.diff(np.where(valid)[0])>1)[0]+1) if len(v)>=3]
    if valid.sum()<12 or not chunks: return None
    x=raw[valid]; z=x/np.median(x)
    diff=np.concatenate([np.diff(raw[c]/np.median(x)) for c in chunks])
    sd=float(z.std()); dsd=float(diff.std()); skew,kurt=moments(z)
    hist=np.histogram(z,bins=16,range=(.25,4))[0].astype(float); hist/=max(hist.sum(),1)
    entropy=-np.sum(hist*np.log(np.maximum(hist,1e-12)))/np.log(16)
    # Correlations computed within contiguous valid runs, weighted by pair count.
    aa=[]
    for lag in [1,2,4]:
        use=[c for c in chunks if len(c)>lag+1]
        aa.append(float(np.average([ac(raw[c]/np.median(x),lag) for c in use],weights=[len(c)-lag for c in use])) if use else 0.)
    return np.r_[np.median(x),np.quantile(x,.75)-np.quantile(x,.25),z.mean(),sd,np.quantile(z,[.05,.25,.75,.95]),
                 np.median(abs(z-np.median(z))),np.sqrt(np.mean(diff**2)),np.mean(abs(diff)),np.quantile(abs(diff),.95),
                 *aa,dsd/np.sqrt(2),np.sqrt(max(2*sd**2-dsd**2/2,0)),skew,kurt,entropy,1-valid.mean()]

def extract(z):
    bags=[]; info={'ppg_records_accepted':0,'ppg_records_rejected':0,'nni_records_accepted':0,'nni_records_rejected':0}
    for name, func in [('ppg',ppg_record),('nni',nni_record)]:
        raw=z[name+'_raw']; offsets=z[name+'_offsets']
        for a,b in zip(offsets[:-1],offsets[1:]):
            f=func(raw[int(a):int(b)])
            if f is None or not np.isfinite(f).all(): info[name+'_records_rejected']+=1; continue
            v=np.full(HEART_DIM,np.nan,float)
            if name=='ppg':v[:PPG_DIM]=f; v[-2:]=[1,0]
            else:v[PPG_DIM:-2]=f; v[-2:]=[0,1]
            bags.append(v); info[name+'_records_accepted']+=1
    # PPG and NNI record types remain separate instances, with no forced visit alignment.
    return np.stack(bags).astype(np.float32) if bags else np.empty((0,HEART_DIM),np.float32),info
