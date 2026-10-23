"""ABMIL pooling, task-specific modality gates, BatchEnsemble-inspired heads.

These are explicit adaptations, not exact reproductions of ABMIL or TabM.
"""
import math,random
import torch
from torch import nn
from features_v171 import VIEWS

class BagEncoder(nn.Module):
    def __init__(self,d,width,p,attention):
        super().__init__();self.attention=attention
        self.phi=nn.Sequential(nn.Linear(d,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.v=nn.Linear(width,width);self.u=nn.Linear(width,width)
        self.a=nn.Linear(width,1,bias=False)
        self.rho=nn.Sequential(nn.Linear(width*2,width),nn.LayerNorm(width),nn.SiLU(),nn.Dropout(p))
    def forward(self,x,mask):
        h=self.phi(x);w=mask.to(h.dtype)
        if self.attention:
            logits=self.a(self.v(h).tanh()*self.u(h).sigmoid()).squeeze(-1)
            w=logits.masked_fill(~mask,-1e4).softmax(1)*w
        w=w/w.sum(1,keepdim=True).clamp_min(1e-8)
        mu=(h*w[:,:,None]).sum(1)
        var=((h-mu[:,None])**2*w[:,:,None]).sum(1)
        return self.rho(torch.cat([mu,var],-1))*mask.any(1)[:,None]

class TypedMIL(nn.Module):
    def __init__(self,dims,summary_dim,schema,width=24,hidden=48,dropout=.3,
                 attention=True,skip=True,modality_dropout=.1,**unused):
        super().__init__();self.schema=schema;self.modality_dropout=modality_dropout
        self.encoders=nn.ModuleDict({v:BagEncoder(2*dims[v],width,dropout,attention) for v in VIEWS})
        self.gates=nn.ModuleList([nn.Linear(width,1) for _ in range(3)])
        self.shared=nn.Sequential(nn.Linear(2*width+4,hidden),nn.LayerNorm(hidden),nn.SiLU(),nn.Dropout(dropout))
        self.indices=[[j for j,t in enumerate(schema) if t['family']==f] for f in ('PSQI','SCL90','EPQ')]
        self.private=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.SiLU(),nn.Dropout(dropout)) for _ in self.indices])
        self.heads=nn.ModuleList([nn.Linear(hidden,len(ix)) for ix in self.indices])
        self.skip=nn.Linear(summary_dim,len(schema),bias=False) if skip else None
        for head in self.heads:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        if self.skip is not None:nn.init.zeros_(self.skip.weight)
    def shared_parameters(self):
        return list(self.encoders.parameters())+list(self.shared.parameters())
    def forward(self,x,m,summary,details=False):
        z=torch.stack([self.encoders[v](x[v],m[v]) for v in VIEWS],1)
        present=torch.stack([m[v].any(1) for v in VIEWS],1)
        if not present.any(1).all():raise ValueError('All views absent')
        # Never drop views when the summary skip would expose those same inputs.
        # Input-level modality ablations are instead trained as separate recipes.
        weights=[];pred=z.new_zeros((len(z),len(self.schema)))
        denom=present.sum(1,keepdim=True).clamp_min(1)
        mu=(z*present[:,:,None]).sum(1)/denom
        for ix,gate,private,head in zip(self.indices,self.gates,self.private,self.heads):
            w=gate(z).squeeze(-1).masked_fill(~present,-1e4).softmax(1)*present
            w=w/w.sum(1,keepdim=True).clamp_min(1e-8);weights.append(w)
            pooled=(z*w[:,:,None]).sum(1)
            h=self.shared(torch.cat([pooled,mu,present.float()],1))
            pred[:,ix]=head(private(h))
        if self.skip is not None:pred=pred+self.skip(summary)
        return (pred,torch.stack(weights,1)) if details else pred

class EnsembleLinear(nn.Module):
    def __init__(self,di,do,k):
        super().__init__();self.weight=nn.Parameter(torch.empty(do,di))
        self.r=nn.Parameter(torch.empty(k,di));self.s=nn.Parameter(torch.ones(k,do))
        self.bias=nn.Parameter(torch.zeros(k,do))
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        with torch.no_grad():self.r.copy_(torch.randint(0,2,self.r.shape)*2-1)
    def forward(self,x):return nn.functional.linear(x*self.r,self.weight)*self.s+self.bias

class TabEnsemble(nn.Module):
    def __init__(self,summary_dim,schema,hidden=48,dropout=.25,members=8,**unused):
        super().__init__();self.members=members
        self.first=EnsembleLinear(summary_dim,hidden,members)
        self.second=EnsembleLinear(hidden,hidden,members)
        with torch.no_grad():self.second.r.fill_(1.)
        self.dropout=nn.Dropout(dropout)
        self.head_weight=nn.Parameter(torch.zeros(members,hidden,len(schema)))
        self.head_bias=nn.Parameter(torch.zeros(members,len(schema)))
        self.skip=nn.Linear(summary_dim,len(schema),bias=False);nn.init.zeros_(self.skip.weight)
    def shared_parameters(self):return list(self.parameters())
    def forward(self,x,m,summary,details=False):
        h=self.dropout(self.first(summary[:,None]).relu())
        h=self.dropout(self.second(h).relu())
        out=torch.einsum('bkh,kht->bkt',h,self.head_weight)+self.head_bias
        return out+self.skip(summary)[:,None]

def family_losses(pred,truth,observed,schema):
    if pred.ndim==2:pred=pred[:,None]
    safe=torch.where(observed,truth,torch.zeros_like(truth))
    losses=nn.functional.smooth_l1_loss(pred,safe[:,None].expand_as(pred),reduction='none',beta=.25)
    counts=observed.sum(0)
    # Mean of member-specific losses, not loss of the mean prediction (TabM).
    per=(losses*observed[:,None]).sum((0,1))/(counts.clamp_min(1)*pred.shape[1])
    terms=[]
    for family in ('PSQI','SCL90','EPQ'):
        weights=pred.new_tensor([t['weight'] if t['family']==family else 0. for t in schema])*(counts>0)
        if weights.sum()>0:terms.append((weights*per).sum()/weights.sum())
    return terms

def pcgrad_backward(losses,model):
    """Project only shared-parameter gradients; private parameters use ordinary mean loss."""
    params=[p for p in model.parameters() if p.requires_grad]
    shared={id(p) for p in model.shared_parameters()}
    shared_ix=[i for i,p in enumerate(params) if id(p) in shared]
    grads=[torch.autograd.grad(l,params,retain_graph=True,allow_unused=True) for l in losses]
    vectors=[torch.cat([(g[i] if g[i] is not None else torch.zeros_like(params[i])).flatten() for i in shared_ix]) for g in grads]
    projected=[];conflicts=0;pairs=0
    for i,g in enumerate(vectors):
        h=g.clone();others=list(range(len(vectors)));random.shuffle(others)
        for j in others:
            if i==j:continue
            dot=torch.dot(h,vectors[j]);pairs+=1
            if dot<0:
                h-=dot/vectors[j].square().sum().clamp_min(1e-12)*vectors[j];conflicts+=1
        projected.append(h)
    total=torch.stack(projected).mean(0);offset=0
    for k,p in enumerate(params):
        if k in shared_ix:
            n=p.numel();p.grad=total[offset:offset+n].view_as(p).clone();offset+=n
        else:
            gg=[g[k] for g in grads if g[k] is not None]
            p.grad=sum(gg)/len(losses) if gg else None
    return conflicts/max(pairs,1)
