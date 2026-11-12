"""Mode-II geometry sweep: fixed-source KM-PIKFNN versus independent BEM.

Run: python modeii_width_height_bem_sweep.py
Requires: numpy scipy matplotlib torch (threadpoolctl recommended).

Full plate: [-b,b] x [-h,h], central crack [-a,a] x {0}.
Quarter-domain loading exactly matches the supplied notebook:
 top: syy=0, sxy=tau0; right: sxx=0, sxy=tau0;
 left: uy=0, sxx=0; crack face: syy=sxy=0;
 ligament: ux=0, syy=0.

Relative width means b/a; relative height means h/a. a is held fixed.
Sources are FIXED at dx/b=0.6, dy/h=0.3, the approximate final source
factors of the earlier (0.5,0.2) cold-start run. This is a baseline choice,
not a claim of global optimality across every geometry. No source search,
source-coordinate optimization, or selection by BEM error is performed.

KM uses an independently computed regularized SVD warm start followed by
coefficient-only L-BFGS. The last physical coefficient is KII.
BEM uses exactly 2304 constant elements / midpoint collocation points on
the quarter boundary (nbase=256), analytical panel integrals and grade=2.
There is NO BEM mesh refinement. Three extraction windows are retained
as diagnostics only; the 0.01 <= r/a <= 0.08 window supplies the reference.
Differences from this finite-resolution BEM are comparison differences,
not errors relative to an exact solution.

Outputs: CSV, Markdown table, comparison/error figures (PNG and PDF),
per-case optimizer histories, settings JSON, independent BEM caches.
Change the lists in CFG, or use --widths / --heights. --plot-only reads
an existing CSV. Training uses CUDA when available; BEM remains CPU.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.linalg import lstsq as scipy_lstsq
import torch

CFG = {
    "a": 1.0, "E": 1.0, "nu": 0.30, "tau0": 1.0, "plane_strain": True,
    "width_ratios_b_over_a": [1.5, 2.0, 2.5, 3.0, 4.0],
    "height_ratios_h_over_a": [1.0, 2.0, 3.0, 4.0, 6.0],
    "source_margin_x_factor": 0.60, "source_margin_y_factor": 0.30,
    "n_src_side": 64, "higher_williams_orders": [1.5, 2.5],
    "n_top": 121, "n_side": 121, "n_crack": 100, "n_ligament": 100,
    "eps_crack": 1e-6, "svd_rcond": 1e-12, "lambda_reg": 1e-10,
    "w_top": 2.5, "w_right": 1.5, "w_left": 4.0,
    "w_crack": 4.0, "w_ligament": 4.0,
    "lbfgs_epochs": 20, "lbfgs_max_iter": 20, "print_every": 5,
    "training_device": "auto", "blas_threads": 1,
    "bem_nbase": 256, "bem_grade": 2.0,
    "plot_normalized_KII": True, "show_figures": True,
    "output_dir": "modeii_width_height_results",
}
QUANTITIES = ("ux", "uy", "sxx", "syy", "sxy")

# Independent direct, constant-element BEM: H u = G t.
# Physical-boundary elements only; no KM-network coefficients or fictitious sources.
def _bem_integral_block(x,c,t,n,L,mu,kappa):
    d=c[None,:,:]-x[:,None,:]
    alpha=np.einsum('ijk,jk->ij',d,t); beta=np.einsum('ijk,jk->ij',d,n)
    s0=alpha-L[None,:]/2;s1=alpha+L[None,:]/2
    cop=np.abs(beta)<1e-13*max(1,float(np.max(L)))
    ab=np.where(cop,1.,abs(beta)); bb=np.where(cop,0.,beta)
    r0=s0*s0+bb*bb;r1=s1*s1+bb*bb
    da=np.arctan2(ab*(s1-s0),ab*ab+s0*s1)
    I0=np.where(cop,0,da/ab); I1=.5*np.log(r1/r0)
    F0=s0*.5*np.log(r0)-s0+np.where(cop,0,ab*np.arctan(s0/ab))
    F1=s1*.5*np.log(r1)-s1+np.where(cop,0,ab*np.arctan(s1/ab))
    logint=F1-F0
    nn=np.where(cop,0,ab*da);tt=L[None,:]-nn;tn=bb*I1
    Tt=np.einsum('ji,jk->jik',t,t)[None,:,:,:]
    Nn=np.einsum('ji,jk->jik',n,n)[None,:,:,:]
    Tn=np.einsum('ji,jk->jik',t,n)[None,:,:,:]
    Nt=np.einsum('ji,jk->jik',n,t)[None,:,:,:]
    eye=np.eye(2)[None,None,:,:]; C=1/(2*np.pi*mu*(kappa+1))
    G=C*(-kappa*logint[:,:,None,None]*eye+tt[:,:,None,None]*Tt+nn[:,:,None,None]*Nn+tn[:,:,None,None]*(Tn+Nt))
    ds=s1/r1-s0/r0
    jtt=.5*(I0-ds);jnn=.5*(I0+ds);jtn=-.5*bb*(1/r1-1/r0)
    H=C*mu*((kappa-1)*(I1[:,:,None,None]*(Tn-Nt)-(bb*I0)[:,:,None,None]*eye)-4*bb[:,:,None,None]*(jtt[:,:,None,None]*Tt+jnn[:,:,None,None]*Nn+jtn[:,:,None,None]*(Tn+Nt)))
    nr,nc=len(x),len(c)
    return (H.transpose(0,2,1,3).reshape(2*nr,2*nc),
            G.transpose(0,2,1,3).reshape(2*nr,2*nc))

def bem_panel_matrices(c,t,n,L,mu,kappa):
    count=len(c)
    H=np.empty((2*count,2*count));G=np.empty_like(H)
    for first in range(0,count,96):
        last=min(first+96,count)
        H[2*first:2*last],G[2*first:2*last]=_bem_integral_block(c[first:last],c,t,n,L,mu,kappa)
    H.flat[::2*count+1]+=.5
    return H,G

def solve_bem_mode_ii(a=1,b=2,h=6,E=1,nu=.3,tau=1,nbase=64,grade=2,plane_strain=True):
    mu=E/(2*(1+nu)); kappa=3-4*nu if plane_strain else (3-nu)/(1+nu)
    p=[]; tags=[]
    def segment(nodes,tag):
        p.extend(zip(nodes[:-1],nodes[1:]));tags.extend([tag]*(len(nodes)-1))
    u=np.linspace(0,1,2*nbase+1)
    segment(np.column_stack((a*(1-(1-u)**grade),0*u)),'crack')
    segment(np.column_stack((a+(b-a)*u**grade,0*u)),'ligament')
    u=np.linspace(0,1,2*nbase+1);segment(np.column_stack((b+0*u,h*u)),'right')
    u=np.linspace(0,1,nbase+1);segment(np.column_stack((b*(1-u),h+0*u)),'top')
    u=np.linspace(0,1,2*nbase+1);segment(np.column_stack((0*u,h*(1-u))),'left')
    p=np.asarray(p);tags=np.asarray(tags);c=p.mean(1);v=p[:,1]-p[:,0];L=np.linalg.norm(v,axis=1);t=v/L[:,None];norm=np.column_stack((t[:,1],-t[:,0]));N=len(c)
    H,G=bem_panel_matrices(c,t,norm,L,mu,kappa)
    uk=np.full((N,2),np.nan);tk=uk.copy()
    tk[tags=='crack']=0; tk[tags=='top']=(tau,0);tk[tags=='right']=(0,tau)
    uk[tags=='left',1]=0;tk[tags=='left',0]=0
    uk[tags=='ligament',0]=0;tk[tags=='ligament',1]=0
    iu=np.isnan(uk).ravel();u0=np.nan_to_num(uk).ravel();t0=np.nan_to_num(tk).ravel()
    M=np.where(iu[None,:],H,-G); rhs=G@t0-H@u0
    rigid_error=float(np.max(abs(H@np.tile([1,0],N))))
    del H,G
    gauge=np.zeros(2*N);ig=2*np.flatnonzero(tags=='left')[len(np.flatnonzero(tags=='left'))//2];gauge[ig]=1
    M=np.vstack((M,gauge));rhs=np.r_[rhs,0]
    scale=np.linalg.norm(M,axis=0);scaled=np.array(M,order='F',copy=True);scaled/=scale
    x=scipy_lstsq(scaled,rhs,cond=1e-12,lapack_driver='gelsy',overwrite_a=True)[0]/scale
    del scaled
    uv=np.where(iu,x,u0).reshape(N,2);tv=np.where(iu,t0,x).reshape(N,2)
    fits=[];fit_counts=[];fit_residuals=[]
    for low,high in [(0.002,.02),(.005,.05),(.01,.08)]:
        mask=(tags=='crack')&((a-c[:,0])/a>=low)&((a-c[:,0])/a<=high)
        if mask.sum()<8: raise ValueError('BEM crack-tip fitting needs at least 8 elements per window')
        r=(a-c[mask,0])/a;mat=np.column_stack((0*r+1,np.sqrt(r),r,r**1.5,r*r))
        cf=np.linalg.lstsq(mat,uv[mask,0],rcond=None)[0]
        fit_counts.append(int(mask.sum()))
        fit_residuals.append(float(np.linalg.norm(mat@cf-uv[mask,0])/max(np.linalg.norm(uv[mask,0]),1e-30)))
        fits.append(2*mu/(kappa+1)*np.sqrt(2*np.pi/a)*cf[1])
    bie_residual=float(np.linalg.norm(M@x-rhs)/max(np.linalg.norm(rhs),1e-30))
    resultant=np.sum(tv*L[:,None],axis=0)
    moment=float(np.sum((c[:,0]*tv[:,1]-c[:,1]*tv[:,0])*L))
    return dict(c=c,tags=tags,u=uv,t=tv,L=L,n=norm,K=np.asarray(fits),
                fit_counts=fit_counts,fit_residuals=fit_residuals,
                KII=float(fits[-1]),bie_residual=bie_residual,rigid_error=rigid_error,
                resultant=resultant,moment=moment,nbase=nbase,n_elements=N)


class KMCase:
    def __init__(self, width_ratio, height_ratio, cfg):
        self.cfg = cfg
        self.a = float(cfg["a"])
        self.b = self.a * width_ratio
        self.h = self.a * height_ratio
        self.mu = cfg["E"] / (2 * (1 + cfg["nu"]))
        self.kappa = 3-4*cfg["nu"] if cfg["plane_strain"] else (3-cfg["nu"])/(1+cfg["nu"])
        tau = cfg["tau0"]
        self.blocks = [
            ("top","syy",0.,"w_top"),("top","sxy",tau,"w_top"),
            ("right","sxx",0.,"w_right"),("right","sxy",tau,"w_right"),
            ("left","uy",0.,"w_left"),("left","sxx",0.,"w_left"),
            ("crack","syy",0.,"w_crack"),("crack","sxy",0.,"w_crack"),
            ("ligament","ux",0.,"w_ligament"),("ligament","syy",0.,"w_ligament"),
        ]
        n=cfg["n_src_side"]
        mx=cfg["source_margin_x_factor"]*self.b
        my=cfg["source_margin_y_factor"]*self.h
        xs=np.linspace(-mx,self.b+mx,n);ys=np.linspace(-my,self.h+my,n)
        self.sources=np.r_[xs-1j*my,xs+1j*(self.h+my),-mx+1j*ys[1:-1],self.b+mx+1j*ys[1:-1]]
        assert np.all((self.sources.real<0)|(self.sources.real>self.b)|
                      (self.sources.imag<0)|(self.sources.imag>self.h))

    def boundary_points(self, validation=False):
        cfg=self.cfg;a,b,h=self.a,self.b,self.h
        def line(lo,hi,n):
            if validation:return lo+(np.arange(n)+.5)*(hi-lo)/n
            return np.linspace(lo,hi,n)
        counts=(177,173,151,149) if validation else (cfg["n_top"],cfg["n_side"],cfg["n_crack"],cfg["n_ligament"])
        nt,ns,nc,nl=counts
        # Matches the uploaded notebook over the default sweep range.
        tip_gap=min(.02*a,.2*(b-a))
        return dict(top=line(0,b,nt)+1j*h,right=b+1j*line(0,h,ns),left=1j*line(0,h,ns),
                    crack=line(.02*a,.98*a,nc)+1j*cfg["eps_crack"]*a,
                    ligament=line(a+tip_gap,b,nl)+0j)


    def km_response(self, z, phi, dphi, ddphi, psi, dpsi):
        stress_sum = 4*dphi.real
        stress_diff = 2*(z.conjugate()*ddphi+dpsi)
        displacement = (self.kappa*phi-z*dphi.conjugate()-psi.conjugate())/(2*self.mu)
        return np.column_stack((displacement.real, displacement.imag,
                                (stress_sum-stress_diff.real)/2,
                                (stress_sum+stress_diff.real)/2,
                                stress_diff.imag/2))

    def complex_point_force_potential_response(self, z, source, force):
        q = z-source
        den = 2*np.pi*(self.kappa+1)
        phi = -force/den*np.log(q) + force/(2*np.pi*self.kappa*(self.kappa+1))
        dphi, ddphi = -force/den/q, force/den/q**2
        psi = self.kappa*force.conjugate()/den*np.log(q) + force*source.conjugate()/den/q
        dpsi = self.kappa*force.conjugate()/den/q - force*source.conjugate()/den/q**2
        return self.km_response(z, phi, dphi, ddphi, psi, dpsi)

    def mode_ii_williams_response(self, z, p=0.5):
        if p < 0.5 or abs((p-0.5)-round(p-0.5)) > 1e-12:
            raise ValueError("Use p=0.5, 1.5, 2.5, ...")
        z = np.asarray(z, dtype=np.complex128).reshape(-1)
        w = z-self.a
        if np.any(np.abs(w) == 0):
            raise ValueError("Do not evaluate exactly at the singular crack tip")
        C = -1j / np.sqrt(2*np.pi) * self.h**(0.5-p)
        phi, dphi = C*w**p, C*p*w**(p-1)
        ddphi = C*p*(p-1)*w**(p-2)
        psi = -(p+1)*phi-self.a*dphi
        dpsi = -(p+1)*dphi-self.a*ddphi
        return self.km_response(z, phi, dphi, ddphi, psi, dpsi)

    def build_basis_columns(self, points, sources=None, uy_sign=-1):
        # uy_sign=+1 deliberately reproduces the OLD, inconsistent leading uy.
        # This flag is diagnostic only. All physical predictions use uy_sign=-1.
        if uy_sign not in (-1, 1):
            raise ValueError("uy_sign must be -1 or +1")
        z = np.asarray(points, dtype=np.complex128).reshape(-1)
        sources = self.sources if sources is None else sources
        cols = [self.complex_point_force_potential_response(z,s,P)
                for s in sources for P in (1+0j, 1j)]
        cols += [self.mode_ii_williams_response(z,p) for p in self.cfg["higher_williams_orders"]]
        leading = self.mode_ii_williams_response(z,0.5)
        if uy_sign == 1:
            leading[:,1] *= -1
        cols.append(leading)
        array = np.stack(cols,axis=2)
        assert np.isfinite(array).all()
        return {q: array[:,i,:] for i,q in enumerate(QUANTITIES)}

    def assemble(self, validation=False):
        points=self.boundary_points(validation)
        basis={g:self.build_basis_columns(z) for g,z in points.items()}
        rows=[];rhs=[]
        for group,q,target,weight in self.blocks:
            block=basis[group][q];factor=np.sqrt(self.cfg[weight]/len(block))
            rows.append(factor*block);rhs.append(np.full(len(block),factor*target))
        return np.vstack(rows),np.concatenate(rhs),basis

    def train(self, device):
        start=time.perf_counter()
        physical,rhs,_=self.assemble()
        scale=np.linalg.norm(physical,axis=0);scale=np.where(scale>0,scale,1.)
        A=physical/scale
        reg=self.cfg["lambda_reg"]
        n=A.shape[1]
        R=np.zeros((n-1,n));R[:,:n-1]=np.sqrt(reg/(n-1))*np.eye(n-1)
        # Same regularized objective as subsequent L-BFGS; KII is unpenalized.
        theta0=scipy_lstsq(np.vstack((A,R)),np.r_[rhs,np.zeros(n-1)],
                          cond=self.cfg["svd_rcond"],lapack_driver="gelsd")[0]
        At=torch.as_tensor(A,dtype=torch.float64,device=device)
        bt=torch.as_tensor(rhs,dtype=torch.float64,device=device)
        theta=torch.nn.Parameter(torch.as_tensor(theta0,dtype=torch.float64,device=device).clone())
        def losses():
            boundary=(At@theta-bt).square().sum()
            return boundary,boundary+reg*theta[:-1].square().mean()
        with torch.no_grad():boundary,total=losses()
        best=theta.detach().clone();best_loss=total.item()
        history=[dict(iteration=0,stage="regularized_SVD_warm_start",KII=float(theta0[-1]/scale[-1]),
                      boundary_loss=boundary.item(),total_loss=best_loss)]
        opt=torch.optim.LBFGS([theta],lr=1.,max_iter=self.cfg["lbfgs_max_iter"],
                             max_eval=self.cfg["lbfgs_max_iter"]+10,history_size=100,
                             tolerance_grad=1e-12,tolerance_change=1e-14,line_search_fn="strong_wolfe")
        for epoch in range(1,self.cfg["lbfgs_epochs"]+1):
            def closure():
                opt.zero_grad();loss=losses()[-1];loss.backward();return loss
            opt.step(closure)
            with torch.no_grad():
                boundary,total=losses()
                bv,tv,kii=torch.stack((boundary,total,theta[-1]/scale[-1])).cpu().tolist()
            history.append(dict(iteration=epoch,stage="L-BFGS",KII=kii,boundary_loss=bv,total_loss=tv))
            if tv<best_loss:best_loss=tv;best=theta.detach().clone()
            if epoch==1 or epoch%self.cfg["print_every"]==0:
                print(f"  L-BFGS {epoch:3d}: KII={kii:.9f}, boundary loss={bv:.3e}, total={tv:.3e}",flush=True)
            # Gradient stopping criterion; never stop based on BEM error.
            opt.zero_grad();losses()[-1].backward()
            if theta.grad.detach().abs().max().item()<1e-12:break
        self.coeff=best.cpu().numpy()/scale
        if device.type=="cuda":torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-start
        field=self.build_basis_columns(np.array([self.a+.001*self.a+0j]))
        assert np.isfinite((field["sxy"]@self.coeff)).all()
        validation,rhs_val,basis_val=self.assemble(validation=True)
        validation_loss=float(np.sum((validation@self.coeff-rhs_val)**2))
        components=[]
        for group,q,target,weight in self.blocks:
            components.append(float(np.mean((basis_val[group][q]@self.coeff-target)**2)))
        return dict(KII=float(self.coeff[-1]),warm_KII=float(theta0[-1]/scale[-1]),
                    train_loss=float(np.sum((physical@self.coeff-rhs)**2)),
                    weighted_validation_loss=validation_loss,boundary_loss_sum=sum(components),
                    elapsed=elapsed,history=history,n_sources=len(self.sources))



def bem_reference_for(case, folder, cfg):
    signature=json.dumps(dict(version="ModeII_direct_BEM_2304_v1",a=case.a,b=case.b,h=case.h,
        E=cfg["E"],nu=cfg["nu"],tau=cfg["tau0"],plane_strain=cfg["plane_strain"],
        nbase=cfg["bem_nbase"],grade=cfg["bem_grade"]),sort_keys=True)
    cache=folder/"bem_cache"/(hashlib.sha256(signature.encode()).hexdigest()+".json")
    if cache.is_file():
        saved=json.loads(cache.read_text())
        if saved["signature"]==signature and saved["result"]["n_elements"]==2304:
            result=saved["result"]
            if np.isfinite(result["KII"]) and result["bie_residual"]<1e-3:
                return {**result,"cache_used":True,"current_elapsed":0.}
    start=time.perf_counter()
    out=solve_bem_mode_ii(case.a,case.b,case.h,cfg["E"],cfg["nu"],cfg["tau0"],
                         cfg["bem_nbase"],cfg["bem_grade"],cfg["plane_strain"])
    assert out["n_elements"]==2304
    if (not np.isfinite(out["KII"]) or abs(out["KII"])<1e-12 or out["bie_residual"]>=1e-3
        or out["rigid_error"]>=1e-9):
        raise RuntimeError("BEM equation/rigid-translation check failed for this geometry")
    result=dict(KII=out["KII"],n_elements=out["n_elements"],bie_residual=out["bie_residual"],
                fit_spread=float(np.ptp(out["K"])/abs(out["KII"])),
                extraction_KII=list(map(float,out["K"])),elapsed=time.perf_counter()-start)
    cache.parent.mkdir(parents=True,exist_ok=True)
    temporary=cache.with_suffix(".tmp");temporary.write_text(json.dumps(dict(signature=signature,result=result),indent=2))
    temporary.replace(cache)
    return {**result,"cache_used":False,"current_elapsed":result["elapsed"]}


def write_csv(path, rows):
    with path.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_table(path, rows):
    lines=["# Mode-II crack: fixed-source KM-PIKFNN versus 2304-element BEM","",
           "Relative width = b/a; relative height = h/a. Fixed source offsets: dx/b=0.6, dy/h=0.3.",
           "Differences are relative to the finite-resolution BEM reference; no BEM mesh refinement is performed.","",
           "| b/a | a/b | h/a | h/b | KII (KM) | KII (BEM) | Difference (%) | Held-out weighted loss | BEM fit spread (%) |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['b_over_a']:g} | {r['a_over_b']:.4f} | {r['h_over_a']:g} | {r['h_over_b']:.4f} | "
                     f"{r['KII_KM']:.8f} | {r['KII_BEM']:.8f} | {r['difference_percent']:.4f} | "
                     f"{r['weighted_validation_loss']:.3e} | {r['bem_fit_spread_percent']:.4f} |")
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def plot_results(rows, folder, cfg):
    plt.rcParams.update({"font.family":"serif","font.size":11,"axes.labelsize":13,
                         "mathtext.fontset":"stix","axes.linewidth":.8})
    colors=["#326A93","#BE4540","#598A51","#9863CF","#D38B35","#4DA5A0","#787878"]
    widths=sorted({r["b_over_a"] for r in rows});heights=sorted({r["h_over_a"] for r in rows})
    normalized=cfg["plot_normalized_KII"]
    km_key,bem_key=("FII_KM","FII_BEM") if normalized else ("KII_KM","KII_BEM")
    ylabel=r"$K_{II}/(\tau_0\sqrt{\pi a})$" if normalized else r"$K_{II}$"
    for error_plot in (False,True):
        fig,axes=plt.subplots(1,2,figsize=(11.4,4.4),dpi=160)
        fig.subplots_adjust(left=.08,right=.98,bottom=.17,top=.77,wspace=.30)
        for panel,(fixed_values,key,xkey,xlabel) in enumerate([
            (heights,"h_over_a","b_over_a",r"$b/a$"),
            (widths,"b_over_a","h_over_a",r"$h/a$")]):
            ax=axes[panel]
            for j,value in enumerate(fixed_values):
                group=sorted([r for r in rows if r[key]==value],key=lambda r:r[xkey])
                x=[r[xkey] for r in group];color=colors[j%len(colors)]
                label=(rf"$h/a={value:g}$" if panel==0 else rf"$b/a={value:g}$")
                if error_plot:
                    ax.plot(x,[r["difference_percent"] for r in group],"o-",lw=1.5,ms=4,
                            color=color,mfc="white",label=label)
                else:
                    ax.plot(x,[r[km_key] for r in group],"-",color=color,lw=1.6,marker=".",ms=4,label=label)
                    ax.plot(x,[r[bem_key] for r in group],"o",color=color,mfc="white",ms=4.5,mew=1)
            ax.set_xlabel(xlabel);ax.set_ylabel("Difference from BEM (%)" if error_plot else ylabel)
            ax.tick_params(direction="in",top=True,right=True);ax.grid(alpha=.16,lw=.5)
            ax.legend(loc="lower center",bbox_to_anchor=(.5,1.02),ncol=3,frameon=False,fontsize=9)
        if not error_plot:
            fig.legend(handles=[Line2D([0],[0],color="black",lw=1.6,label="CV-PIKFNN"),
                Line2D([0],[0],color="black",ls="",marker="o",mfc="white",label="BEM (2304 elements)")],
                loc="upper center",bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
        stem="kii_relative_difference" if error_plot else "kii_width_height_comparison"
        fig.savefig(folder/(stem+".png"),dpi=600,bbox_inches="tight")
        fig.savefig(folder/(stem+".pdf"),bbox_inches="tight")
        if cfg["show_figures"]:plt.show()
        else:plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--widths",type=float,nargs="+",help="b/a values")
    parser.add_argument("--heights",type=float,nargs="+",help="h/a values")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--device",choices=["auto","cpu","cuda"])
    parser.add_argument("--no-show",action="store_true")
    parser.add_argument("--plot-only",action="store_true")
    args=parser.parse_args();cfg=CFG.copy()
    if args.widths:cfg["width_ratios_b_over_a"]=args.widths
    if args.heights:cfg["height_ratios_h_over_a"]=args.heights
    if args.output:cfg["output_dir"]=str(args.output)
    if args.device:cfg["training_device"]=args.device
    if args.no_show:cfg["show_figures"]=False
    folder=Path(cfg["output_dir"]);folder.mkdir(parents=True,exist_ok=True)
    if args.plot_only:
        with (folder/"kii_geometry_comparison.csv").open(encoding="utf-8-sig",newline="") as stream:
            rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
        plot_results(rows,folder,cfg);return
    assert cfg["a"]>0 and cfg["E"]>0 and -.99<cfg["nu"]<.5 and cfg["tau0"]!=0
    assert all(np.isfinite(v) and v>1 for v in cfg["width_ratios_b_over_a"])
    assert all(np.isfinite(v) and v>0 for v in cfg["height_ratios_h_over_a"])
    assert cfg["n_src_side"]>=2 and cfg["lambda_reg"]>=0
    assert cfg["bem_nbase"]==256,"Keep nbase=256: exactly 2304 BEM elements are required"
    assert cfg["source_margin_x_factor"]>0 and cfg["source_margin_y_factor"]>0
    requested=cfg["training_device"]
    if requested=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA is unavailable in this Python environment")
    device=torch.device("cuda" if requested=="auto" and torch.cuda.is_available() else "cpu" if requested=="auto" else requested)
    print(f"KM coefficient training: {device}, float64, regularized-SVD warm start + L-BFGS")
    print("Fixed source factors:",cfg["source_margin_x_factor"],cfg["source_margin_y_factor"])
    print("BEM: 2304 quarter-boundary elements; no mesh refinement",flush=True)
    (folder/"settings.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    try:
        from threadpoolctl import threadpool_limits
        context=threadpool_limits(limits=cfg["blas_threads"])
    except ImportError:context=nullcontext()
    rows=[]
    with context:
        for wr in sorted(set(cfg["width_ratios_b_over_a"])):
            for hr in sorted(set(cfg["height_ratios_h_over_a"])):
                print(f"\nb/a={wr:g}, h/a={hr:g}",flush=True)
                case=KMCase(wr,hr,cfg)
                km=case.train(device)
                bem=bem_reference_for(case,folder,cfg)
                denom=cfg["tau0"]*np.sqrt(np.pi*case.a)
                row=dict(b_over_a=wr,a_over_b=1/wr,h_over_a=hr,h_over_b=hr/wr,
                         a=case.a,b=case.b,h=case.h,KII_KM=km["KII"],KII_BEM=bem["KII"],
                         FII_KM=km["KII"]/denom,FII_BEM=bem["KII"]/denom,
                         difference_percent=100*abs(km["KII"]-bem["KII"])/abs(bem["KII"]),
                         alpha_x=cfg["source_margin_x_factor"],alpha_y=cfg["source_margin_y_factor"],
                         n_sources=km["n_sources"],bem_elements=bem["n_elements"],
                         train_boundary_loss=km["train_loss"],weighted_validation_loss=km["weighted_validation_loss"],
                         boundary_loss_sum=km["boundary_loss_sum"],bem_fit_spread_percent=100*bem["fit_spread"],
                         bem_bie_residual=bem["bie_residual"],KM_time_s=km["elapsed"],
                         BEM_original_time_s=bem["elapsed"],BEM_current_time_s=bem["current_elapsed"],
                         bem_cache_used=int(bem["cache_used"]))
                rows.append(row)
                print(f"  KM={row['KII_KM']:.9f}, BEM={row['KII_BEM']:.9f}, "
                      f"difference={row['difference_percent']:.4f}%, held-out loss={row['weighted_validation_loss']:.2e}, "
                      f"BEM fit spread={row['bem_fit_spread_percent']:.3f}%",flush=True)
                write_csv(folder/"kii_geometry_comparison.csv",rows)
                write_table(folder/"kii_geometry_comparison.md",rows)
                write_csv(folder/f"optimizer_ba_{wr:g}_ha_{hr:g}.csv",km["history"])
    plot_results(rows,folder,cfg)
    print(f"\nCompleted {len(rows)} cases. Results: {folder.resolve()}",flush=True)

if __name__=="__main__":main()
