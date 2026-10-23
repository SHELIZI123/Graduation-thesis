"""Low-capacity residual fusion with explicit unimodal regression supervision.

MMPareto-inspired adaptation to masked Huber regression/AdamW, not its original
classification/SGD benchmark. No statistical bypass is jointly optimized.
"""
import torch
from torch import nn
from features_v171 import FOCUS

class ResidualFusion(nn.Module):
    def __init__(self,dims,schema,width=16):
        super().__init__();self.schema=schema
        self.encoders=nn.ModuleList([nn.Sequential(nn.Linear(d,width),nn.LayerNorm(width),nn.SiLU(),nn.Dropout(.25)) for d in dims])
        self.fusion=nn.Sequential(nn.Linear(3*width+3,32),nn.SiLU(),nn.Dropout(.25),nn.Linear(32,len(schema)))
        self.aux=nn.ModuleList([nn.Linear(width,len(schema)) for _ in range(3)])
        for h in [self.fusion[-1],*self.aux]:
            nn.init.normal_(h.weight,std=.01);nn.init.zeros_(h.bias)
    def forward(self,x,present):
        z=[e(v)*present[:,k:k+1] for k,(e,v) in enumerate(zip(self.encoders,x))]
        fused=self.fusion(torch.cat([*z,present.float()],1))
        return fused,[h(v) for h,v in zip(self.aux,z)]

def masked_loss(pred,truth,mask,schema):
    error=nn.functional.smooth_l1_loss(pred,truth,reduction='none',beta=.25)
    counts=mask.sum(0);per=(error*mask).sum(0)/counts.clamp_min(1)
    terms=[]
    for f in ['PSQI','SCL90','EPQ']:
        weights=pred.new_tensor([(1. if t['name'] in FOCUS else .05) if t['family']==f else 0 for t in schema])*(counts>0)
        if weights.sum()>0:terms.append((weights*per).sum()/weights.sum())
    return torch.stack(terms).mean() if terms else pred.sum()*0

def pareto_pair(g,h,gamma=1.5):
    """Two-gradient MMPareto direction and magnitude, with zero-safe branches."""
    dot=torch.dot(g,h);ng=g.norm();nh=h.norm();cos=dot/(ng*nh).clamp_min(1e-12)
    if ng<1e-12 or nh<1e-12:return g+h,float(cos),1.
    if dot>=0:v=g+h
    else:
        d=g-h;alpha=((h@h-g@h)/(d@d).clamp_min(1e-12)).clamp(0,1)
        v=2*(alpha*g+(1-alpha)*h)
    nv=v.norm()
    if nv<1e-12:return v,float(cos),0.
    magnitude=gamma*(g+h).norm();ratio=float(magnitude/nv)
    return v*(magnitude/nv),float(cos),ratio

def update_gradients(fused_loss,aux_losses,model,variant):
    weight=.1 # .3 mean over the three modality auxiliary losses
    replacements=[];logs=[]
    if variant=='mmpareto':
        for enc,aux in zip(model.encoders,aux_losses):
            params=list(enc.parameters())
            gg=torch.autograd.grad(fused_loss,params,retain_graph=True,allow_unused=True)
            hh=torch.autograd.grad(weight*aux,params,retain_graph=True,allow_unused=True)
            g=torch.cat([(a if a is not None else torch.zeros_like(p)).flatten() for a,p in zip(gg,params)])
            h=torch.cat([(a if a is not None else torch.zeros_like(p)).flatten() for a,p in zip(hh,params)])
            v,cos,ratio=pareto_pair(g,h);replacements.append((params,v));logs.append(dict(cosine=cos,rescale=ratio))
    total=fused_loss+(weight*sum(aux_losses) if variant!='plain' else 0)
    total.backward()
    for params,v in replacements:
        offset=0
        for p in params:
            p.grad=v[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
    return logs
