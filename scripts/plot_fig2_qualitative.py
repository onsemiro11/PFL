#!/usr/bin/env python3
"""Fig. 2 from genuine, matched fair-evaluator outputs (no training).

Run with /home/hyundo/miniconda3/envs/epn/bin/python scripts/plot_fig2_qualitative.py
The explicit population is the existing prediction-blind 512-scene test cache,
NOT the complete 1,320,103-scene test split. This restriction is disclosed in
all output provenance and the caption. K=6 is the evaluator's first six of K=20.
Interaction thresholds are unchanged. --selection median reproduces the original
representative selection. The default pfl_advantage mode explicitly selects
favourable examples and must not be interpreted as an aggregate performance claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch
from matplotlib.text import Text
from experiments.common import (METRES_PER_FOOT as FT, dataset_paths,
    full_horizon_subset, gt_free_mean, gt_free_medoid, oracle_endpoint,
    seed_everything, target_metadata, write_csv, atomic_json, load_config)
from experiments.scripts.evaluate_model import build_pfl, epn_args, load_verified, move, unpack

OUT = ROOT / 'figures'
CACHE = ROOT / 'experiments/results/qualitative/cache'
CONFIG = {'pfl': ROOT/'experiments/configs/pfl_stage2.yaml',
          'epn': ROOT/'experiments/configs/epn.yaml'}
CKPT = {'pfl': ROOT/'PFL/trained_models/stage2_ngsim_seed0/stage2_ngsim_seed0-best.tar',
        'epn': ROOT/'EPN/trained_models/fair_ngsim_epn_epn_seed0/fair_ngsim_epn_epn_seed0-best.tar'}
SEED = 0
COLORS = dict(ego='#E69F00', target='#0072B2', other='#999999',
              candidates='#56B4E9', mean='#D62728', medoid='#006D3B', oracle='#542788')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def key(m):
    return tuple(int(m[x]) for x in ('dataset_segment','ego_vehicle_id','vehicle_id','frame_id'))


def load_cache(model):
    path = CACHE/f'ngsim_{model}_seed0.npz'
    if not path.exists():
        subprocess.run([sys.executable, str(ROOT/'experiments/scripts/collect_qualitative.py'),
            '--model', model, '--dataset', 'ngsim', '--config', str(CONFIG[model]),
            '--checkpoint', str(CKPT[model]), '--seed', '0', '--scene-count', '512',
            '--batch-size', '16', '--output', str(path)], check=True, cwd=ROOT)
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d['metadata'] = json.loads(str(d['metadata_json']))
    d['protocol'] = json.loads(str(d['protocol_json']))
    assert Path(d['protocol']['checkpoint']).resolve() == CKPT[model].resolve()
    assert d['protocol']['seed'] == SEED and d['protocol']['K'] == 20
    assert d['protocol']['scene_sampling'] == 'linspace_over_ordered_test_set'
    manifest = json.loads((CKPT[model].parent/'run_manifest.json').read_text())
    assert manifest['config_sha256'] == digest(CONFIG[model])
    assert manifest['dataset'] == 'ngsim' and manifest['seed'] == SEED
    assert manifest['eligible_for_fair_summary']
    return d


def dataset(model='pfl'):
    sys.path.insert(0, str(ROOT/model.upper()))
    from data import highwayTrajDataset
    return highwayTrajDataset(str(dataset_paths('ngsim')['test']), targ_enc_size=112,
        grid_size=[25,5], fit_plan_traj=True, fit_plan_further_ds=1)


def verify_inference(model, indices):
    """Replay original complete cache batches, preserving RNG and batch layout."""
    d = load_cache(model)
    seed_everything(SEED, deterministic=False)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(CONFIG[model])
    if model == 'pfl':
        net, _, transform = build_pfl(cfg, device)
        load_verified(net, CKPT[model], CONFIG[model], 'ngsim', SEED, transform)
    else:
        sys.path.insert(0, str(ROOT/'EPN'))
        from model import EPNet
        net = EPNet(epn_args(cfg)).to(device)
        load_verified(net, CKPT[model], CONFIG[model], 'ngsim', SEED)
        net.train_output_flag = False
    ds = dataset(model)
    net.eval()
    target_lookup = {key(m): i for i,m in enumerate(d['metadata'])}
    scene_lookup = {(ds.itsDsId(int(s)), ds.itsPlanVehId(int(s)), int(ds.itsTime(int(s)))): j
                    for j,s in enumerate(d['scene_indices'])}
    batches = set()
    for i in indices:
        seg, ego, _, frame = key(d['metadata'][i])
        batches.add(scene_lookup[(seg,ego,frame)]//16)
    maxerr = 0.
    with torch.no_grad():
        for b in sorted(batches):
            ids = d['scene_indices'][b*16:(b+1)*16]
            raw = ds.collate_fn([ds[int(s)] for s in ids])
            enc,hist,future,mask,idxs = unpack(move(raw,device))
            metas = target_metadata(ds,idxs,'ngsim',30.)
            ctx = net.encode(*enc,None) if model == 'pfl' else net.soc_encode(*enc,None)
            torch.manual_seed(SEED*1000003+b)
            samples = (net.sample(ctx,hist,n_steps=10,n_samples=20,chunk=20)
                       if model == 'pfl' else net.generate_trajectories(ctx,20)[0][...,:2])
            actual = samples.permute(1,0,2,3).cpu().numpy()
            wanted = d['samples'][[target_lookup[key(m)] for m in metas]]
            np.testing.assert_allclose(actual,wanted,rtol=2e-5,atol=2e-4)
            maxerr = max(maxerr,float(np.max(np.abs(actual-wanted))))
            print(f'{model}: replay batch {b}, max sample difference {maxerr:.8g} ft',flush=True)
    atomic_json(OUT/f'fig02_{model}_cache_verification.json',
        dict(model=model,batches=sorted(batches),max_absolute_difference_ft=maxerr,
             checkpoint_sha256=digest(CKPT[model]),cache_sha256=digest(CACHE/f'ngsim_{model}_seed0.npz')))


def estimates(d):
    # Same selectors and mask as the fair evaluator. maskedMSETest is the existing
    # EPN evaluator metric kernel; aggregate its squared errors over all 25 states.
    # This is trajectory RMSE, not the fair summary's endpoint-only RMSE at 5 s.
    from utils import maskedMSETest
    samples = torch.from_numpy(d['samples'][:,:6]).permute(1,0,2,3)
    future = torch.from_numpy(d['future']).permute(1,0,2)
    mask = torch.from_numpy(d['future_mask']).permute(1,0,2)
    ss,ff,mm,metadata = full_horizon_subset(samples,future,mask,d['metadata'])
    keep = mask[-1,:,0].bool().numpy()
    assert torch.all(mm == 1), 'Endpoint mask did not imply a complete future'
    assert all(torch.isfinite(t).all() for t in (ss,ff,mm))
    preds = dict(mean=gt_free_mean(ss),medoid=gt_free_medoid(ss),oracle=oracle_endpoint(ss,ff[-1]))
    metrics = {}
    for name,pred in preds.items():
        sq,count = maskedMSETest(pred.permute(1,0,2),ff,mm,separately=True)
        rmse = (sq.sum(0)/count.sum(0)).sqrt().numpy()*FT
        # Independent calculation in the plotted metre coordinate system.
        plotted = pred.numpy().astype(np.float64)*FT
        gt = ff.permute(1,0,2).numpy().astype(np.float64)*FT
        np.testing.assert_allclose(rmse,np.sqrt(np.mean(np.sum((plotted-gt)**2,-1),-1)),rtol=2e-6,atol=2e-6)
        metrics[name] = rmse
    return keep, {n:p.numpy()*FT for n,p in preds.items()},metrics


def select(d, keep, metrics, mode):
    from encoder import cv_baseline
    hist = torch.from_numpy(d['target_hist'][keep]).permute(1,0,2)
    # Exactly the fair CV baseline: recent 0.6 s mean displacement, lateral gate=0.
    cv = cv_baseline(hist,25,torch.tensor([0.,1.])).numpy()*FT
    plan = d['ego_plan'][keep]*FT
    cur = d['ego_current'][keep]*FT
    dist = np.linalg.norm(plan-cv,axis=-1)
    minimum = dist.min(-1)
    score = 1./(1.+minimum)  # metres; monotonic inverse closest approach, no model/GT.
    threshold = float(np.quantile(score,.75))
    strong = score >= threshold
    lateral_excursion = np.maximum(np.abs(plan[:,:,0]-cur[:,None,0]).max(-1),
                                   np.abs(d['future'][keep,:,0]*FT).max(-1))
    # A 1.5 m excursion identifies visible lateral movement even when the dataset's
    # +/-4 s manoeuvre label describes a lane change already finished before t=0.
    lateral = lateral_excursion >= 1.5
    eh = d['ego_hist'][keep]*FT
    ego_cv = cv_baseline(torch.from_numpy(eh).permute(1,0,2),25,
                         torch.tensor([0.,1.])).numpy()+cur[:,None,:]
    acceleration_effect = np.abs(plan[:,-1,1]-ego_cv[:,-1,1])
    gap_change = np.ptp(plan[:,:,1]-cv[:,:,1],axis=1)
    # Non-lateral scenes with >=2 m departure from ego CV and >=5 m longitudinal
    # relative-gap range: an observable acceleration/deceleration proxy, not causality.
    longitudinal = (~lateral)&(acceleration_effect>=2.)&(gap_change>=5.)
    delta = metrics['epn']['medoid']-metrics['pfl']['medoid']
    mean_gain = metrics['epn']['mean']-metrics['pfl']['mean']
    joint_gain = np.minimum(mean_gain,delta)
    # Explicit post-hoc illustrative selector, requested by the user. Maximizing
    # the smaller gain requires BOTH GT-free representatives to improve. GT is
    # used to select scenes, never to compute the sample mean or medoid.
    reason = ('largest minimum of mean and medoid EPN-minus-PFL trajectory RMSE gains; '
              'both must be positive; intentionally favourable illustration, not representative'
              if mode == 'pfl_advantage' else
              'minimum absolute distance to category median EPN-minus-PFL medoid trajectory RMSE; no win filter')
    rng = np.random.default_rng(SEED)
    tie_rank = np.empty(len(delta),int)
    tie_rank[rng.permutation(len(delta))] = np.arange(len(delta))
    selected, summary = [], {}
    for name,category in [('longitudinal',longitudinal),('lateral',lateral)]:
        pool = np.flatnonzero(strong&category)
        assert len(pool), f'No {name} scenes; do not silently relax eligibility'
        median = float(np.median(delta[pool]))
        eligible = pool[joint_gain[pool] > 0] if mode == 'pfl_advantage' else pool
        assert len(eligible), f'No genuine PFL joint wins in {name}'
        order = -joint_gain[eligible] if mode == 'pfl_advantage' else np.abs(delta[eligible]-median)
        i = int(eligible[np.lexsort((tie_rank[eligible],order))[0]])
        selected.append((name,i))
        summary[name] = dict(count=len(pool),eligible_count=len(eligible),selected_joint_gain_m=float(joint_gain[i]),median_epn_minus_pfl_medoid_rmse_m=median,
            selected_distance_from_median_m=float(abs(delta[i]-median)),
            pfl_mean_win_fraction=float(np.mean(metrics['pfl']['mean'][pool]<metrics['epn']['mean'][pool])),
            pfl_medoid_win_fraction=float(np.mean(delta[pool]>0)),
            epn_oracle_win_fraction=float(np.mean(metrics['epn']['oracle'][pool]<metrics['pfl']['oracle'][pool])))
    audit = []
    for j,old in enumerate(np.flatnonzero(keep)):
        m=d['metadata'][old]
        row=dict(cache_index=int(old),**m,interaction_score=float(score[j]),
            minimum_ego_plan_target_cv_distance_m=float(minimum[j]),top_quartile=bool(strong[j]),
            lateral_excursion_m=float(lateral_excursion[j]),
            ego_acceleration_effect_m=float(acceleration_effect[j]),longitudinal_gap_range_m=float(gap_change[j]),
            category='lateral' if lateral[j] else ('longitudinal' if longitudinal[j] else 'other'),
            epn_minus_pfl_medoid_rmse_m=float(delta[j]),joint_mean_medoid_gain_m=float(joint_gain[j]),
            selection_mode=mode,selection_reason=reason,seed=SEED)
        for model in metrics:
            for rule in metrics[model]:
                row[f'{model}_{rule}_rmse_m']=float(metrics[model][rule][j])
        audit.append(row)
    return selected,audit,dict(interaction_75_percentile_threshold=threshold,
                              strong_instances=int(strong.sum()),selection_mode=mode,selection_reason=reason,categories=summary)


def geometry(ds, m, d, old):
    seg,ego,target,t = key(m)
    origin = ds.getLocation(seg,target,t)
    np.testing.assert_allclose(ds.getHistory(seg,target,target,t),d['target_hist'][old],atol=1e-4)
    np.testing.assert_allclose(ds.getFuture(seg,target,t),d['future'][old],atol=1e-4)
    np.testing.assert_allclose(ds.getPlanFuture(seg,ego,target,t)[::-1],d['ego_plan'][old],atol=1e-4)
    np.testing.assert_allclose(ds.getHistory(seg,ego,target,t),d['ego_hist'][old],atol=1e-4)
    # NGSIM local Y already increases with traffic. Reflect s only if history says
    # otherwise. This is a shared rigid reflection, preserving every metric.
    direction = 1. if d['target_hist'][old,-1,1]>=d['target_hist'][old,0,1] else -1.
    transform=np.array([1.,direction])
    vehicles=[]
    around=set(ds.getGrid(seg,target,t).tolist())|set(ds.getGrid(seg,ego,t).tolist())|{ego,target}
    for v in sorted(around-{0}):
        tr=ds.Tracks[seg-1][v-1]; col=ds.getTracksCol(seg,v,t)
        if col<0 or col>=tr.shape[1] or tr[0,col]!=t: continue
        center=(tr[1:3,col]-origin)*FT*transform
        h=(tr[1:3,col]-tr[1:3,max(0,col-ds.d_s)])*transform
        if np.linalg.norm(h)<1e-5:
            # Use the latest actual nonzero track displacement for a stopped car.
            diffs=np.diff(tr[1:3,max(0,col-30):col+1],axis=1).T*transform
            nz=diffs[np.linalg.norm(diffs,axis=1)>1e-5]
            h=nz[-1] if len(nz) else np.array([0.,1.])
        vehicles.append(dict(id=v,center=center,heading=h/np.linalg.norm(h),size=ds.getSize(seg,v)*FT,
                             role='ego' if v==ego else ('target' if v==target else 'other')))
    # There is no surveyed lane geometry/width field. Reconstruct only internal
    # boundaries supported by adjacent lane IDs: local lane centres are robust
    # medians of recorded positions; their separation estimates lane width.
    # Use the selected longitudinal corridor across recording time, not invented
    # ramps/curves/shoulders. Exclude merged US101 auxiliary lane 6 and I80 lane 7.
    all_y=np.r_[d['target_hist'][old,:,1],d['future'][old,:,1],d['ego_hist'][old,:,1],d['ego_plan'][old,:,1]]
    lo,hi=origin[1]+all_y.min()-10/FT,origin[1]+all_y.max()+10/FT
    lanes={}
    maxlane=5 if seg<=3 else 6
    for tr in ds.Tracks[seg-1]:
        if tr.ndim!=2 or tr.shape[1]==0: continue
        good=(tr[2]>=lo)&(tr[2]<=hi)&(tr[3]>=1)&(tr[3]<=maxlane)&(tr[4]==1)
        for lane in np.unique(tr[3,good]).astype(int):
            lanes.setdefault(lane,[]).append(tr[1,good&(tr[3]==lane)])
    centers={int(l):float(np.median(np.concatenate(xs))) for l,xs in lanes.items() if sum(map(len,xs))>=20}
    boundaries=[]; widths=[]
    for l in sorted(centers):
        if l+1 in centers:
            width=(centers[l+1]-centers[l])*FT
            if width<=0: continue
            boundaries.append(((centers[l]+centers[l+1])/2-origin[0])*FT)
            widths.append(width)
    assert boundaries, 'No supported internal lane boundary reconstruction'
    return vehicles,boundaries,transform,dict(method='local lane-ID centre medians; adjacent-centre midpoint boundaries',
        surveyed_geometry=False,lane_centers_absolute_ft=centers,inferred_lane_widths_m=widths,
        local_longitudinal_corridor_ft=[float(lo),float(hi)])


def car(ax,v):
    """Small BEV location glyph; shape ratio uses measured dimensions.

    It is deliberately drawn in points, not as a metric footprint: road axes
    exaggerate lateral distances. Caption explicitly discloses this convention.
    """
    from matplotlib.offsetbox import DrawingArea, AnnotationBbox
    from matplotlib.transforms import Affine2D
    from matplotlib.patches import Rectangle
    c,h=v['center'],v['heading']
    length,width=np.asarray(v['size'])*1.7
    pad=max(length,width)+3
    area=DrawingArea(2*pad,2*pad,0,0)
    angle=np.degrees(np.arctan2(h[1],h[0]))
    trans=Affine2D().rotate_deg(angle).translate(pad,pad)+area.get_transform()
    color=COLORS[v['role']]
    body=Rectangle((-length/2,-width/2),length,width,facecolor=color,
                   edgecolor='#444444',lw=.5,alpha=.85,transform=trans)
    area.add_artist(body)
    windshield=Rectangle((length*.15,-width*.38),length*.16,width*.76,
                         facecolor='white',edgecolor='#444444',lw=.3,alpha=.8,transform=trans)
    area.add_artist(windshield)
    ax.add_artist(AnnotationBbox(area,c,frameon=False,pad=0,box_alignment=(.5,.5),zorder=9))


def draw(ds,d,keep,preds,metrics,selected,audit):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
        'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
    fig,axs=plt.subplots(2,2,figsize=(174/25.4,132/25.4),layout='constrained')
    fig.set_constrained_layout_pads(w_pad=.03,h_pad=.04,wspace=.05,hspace=.04)
    geometry_report=[]
    for row,(category,i) in enumerate(selected):
        old=int(np.flatnonzero(keep)[i]); m=d['metadata'][old]
        vehicles,boundaries,transform,geo=geometry(ds,m,d,old)
        gt=d['future'][old]*FT*transform
        h=d['target_hist'][old]*FT*transform
        eh=d['ego_hist'][old]*FT*transform
        plan=d['ego_plan'][old]*FT*transform
        cur=d['ego_current'][old]*FT*transform
        samples={model:load_cache(model)['samples'][old,:6]*FT*transform for model in ('pfl','epn')}
        # Reference-style horizontal road view: a shared axis permutation is an
        # isometry and does not change predictions, selection, or RMSE.
        gt,h,eh,plan,cur=[a[..., ::-1] for a in (gt,h,eh,plan,cur)]
        samples={model:a[..., ::-1] for model,a in samples.items()}
        for v in vehicles:
            v['center']=v['center'][::-1]
            v['heading']=v['heading'][::-1]
        points=np.concatenate([gt,h,eh,plan,*[s.reshape(-1,2) for s in samples.values()]])
        xmin,xmax=points[:,0].min()-5.,points[:,0].max()+7.
        ymin,ymax=min(points[:,1].min()-2.,-4.),max(points[:,1].max()+2.,4.)
        # Shared data limits and axes boxes; lateral enlargement is disclosed.
        for col,model in enumerate(('pfl','epn')):
            ax=axs[row,col]
            for b in boundaries:
                if ymin<=b<=ymax: ax.axhline(b,color='#C8C8C8',lw=.5,ls=(0,(6,6)),zorder=0)
            for v in vehicles:
                if xmin<v['center'][0]<xmax and ymin<v['center'][1]<ymax:
                    if v['role']=='other':
                        seg,ego,target,frame=key(m)
                        other_hist=(ds.getHistory(seg,v['id'],target,frame)*FT*transform)[..., ::-1]
                        ax.plot(other_hist[:,0],other_hist[:,1],color='#AAAAAA',lw=.5,alpha=.75,zorder=1)
                    car(ax,v)
            ax.plot(h[:,0],h[:,1],color=COLORS['target'],lw=.65,zorder=3)
            ax.plot(eh[:,0],eh[:,1],color=COLORS['ego'],lw=.65,zorder=3)
            ax.plot(plan[:,0],plan[:,1],color=COLORS['ego'],lw=1.3,zorder=5)
            ax.plot(gt[:,0],gt[:,1],color='black',lw=1.3,zorder=6)
            for candidate in samples[model]:
                ax.plot(candidate[:,0],candidate[:,1],color=COLORS['candidates'],alpha=.45,lw=.75,zorder=2)
            for name,ls,marker in [('mean','-','o'),('medoid','-','s'),('oracle','--','D')]:
                p=(preds[model][name][i]*transform)[..., ::-1]
                # Final plotted-coordinate assertion including direction conversion.
                observed=np.sqrt(np.mean(np.sum((p.astype(float)-gt.astype(float))**2,-1)))
                np.testing.assert_allclose(observed,metrics[model][name][i],rtol=1e-5,atol=2e-6)
                ax.plot(p[:,0],p[:,1],color=COLORS[name],lw=1.15,ls=ls,zorder=7)
                ax.plot(*p[-1],marker=marker,ms=3,color=COLORS[name],mfc='white',mew=.7,zorder=8)
            title=('Longitudinal' if row==0 else 'Lateral')+' interaction'
            ax.set_title(f'({chr(97+row*2+col)}) '+('PFL two-stage' if model=='pfl' else 'EPN')+'\n'+title,pad=5)
            ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel='Driving $+s$ (m) →',ylabel='$d$ (m)')
            ax.set_aspect('auto')
            ax.grid(False); ax.spines[['top','right']].set_visible(False)
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.set_xticks([x for x in ax.get_xticks() if xmin <= x <= xmax])
            ax.set_yticks([y for y in ax.get_yticks() if ymin <= y <= ymax])
            values=metrics[model]
            ax.text(.5,-.37,
                f"RMSE (m): mean {values['mean'][i]:.2f} | medoid {values['medoid'][i]:.2f}\n"
                f"endpoint-oracle {values['oracle'][i]:.2f}",
                transform=ax.transAxes,ha='center',va='top',fontsize=8)
        geo.update(scene_key=list(key(m)),axis_limits_m=[xmin,xmax,ymin,ymax],
                   horizontal_axis='longitudinal_s',vertical_axis='lateral_d',
                   vehicle_rendering='fixed display scale; measured dimension ratios; not metric footprints')
        geometry_report.append(geo)
    handles=[Patch(color=COLORS['ego'],alpha=.7,label='Ego'),
        Patch(color=COLORS['target'],alpha=.7,label='Target'),
        Patch(color=COLORS['other'],alpha=.4,label='Other vehicles'),
        Line2D([],[],color=COLORS['ego'],lw=1.3,label='Ego plan'),
        Line2D([],[],color='black',lw=1.3,label='Target GT'),
        Line2D([],[],color=COLORS['candidates'],alpha=.6,lw=.75,label='Six candidates'),
        Line2D([],[],color=COLORS['mean'],lw=1.15,marker='o',ms=3,label='Sample mean'),
        Line2D([],[],color=COLORS['medoid'],lw=1.15,marker='s',ms=3,label='Sample medoid'),
        Line2D([],[],color=COLORS['oracle'],lw=1.15,ls='--',label='Oracle (analysis only)')]
    # A reserved layout row accommodates the one figure-wide legend, avoiding
    # bbox_inches=tight changing the requested 174 mm publication width.
    fig.get_layout_engine().set(rect=(0,.23,1,.77))
    legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.018),
                     ncol=3,frameon=False,columnspacing=1.1,handlelength=2.)
    fig.text(.5,.19,('Selected PFL advantage cases; not representative.\n' if audit[0]['selection_mode']=='pfl_advantage' else '')+'Thin lines: histories. Lanes: reconstructed. Vehicles: not to scale.',
             ha='center',va='center',fontsize=8)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer(); canvas=fig.bbox
    for obj in fig.findobj(Text):
        if obj.get_visible() and obj.get_text():
            assert obj.get_fontsize()>=8
            box=obj.get_window_extent(renderer)
            assert box.x0>=-1 and box.y0>=-1 and box.x1<=canvas.x1+1 and box.y1<=canvas.y1+1, (obj.get_text(),box)
    for row in range(2):
        a,b=axs[row]
        np.testing.assert_equal(a.get_xlim(),b.get_xlim());np.testing.assert_equal(a.get_ylim(),b.get_ylim())
        np.testing.assert_allclose(a.get_position().bounds[2:],b.get_position().bounds[2:],atol=1e-8)
        np.testing.assert_allclose(a.transData.get_matrix()[:2,:2],b.transData.get_matrix()[:2,:2],atol=1e-8)
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'fig02_pfl_epn_qualitative.{ext}',dpi=600,facecolor='white')
    plt.close(fig)
    return geometry_report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection',choices=['median','pfl_advantage'],default='pfl_advantage')
    ap.add_argument('--verify-model',choices=['pfl','epn'])
    ap.add_argument('--indices',type=int,nargs='+')
    ap.add_argument('--skip-inference-verification',action='store_true',help='Only for layout iteration after successful verification')
    args=ap.parse_args();OUT.mkdir(exist_ok=True)
    if args.verify_model:
        verify_inference(args.verify_model,args.indices);return
    seed_everything(SEED,deterministic=False);torch.set_num_threads(4)
    before={m:digest(p) for m,p in CKPT.items()}
    data={m:load_cache(m) for m in ('pfl','epn')};d=data['pfl']
    assert [key(m) for m in d['metadata']]==[key(m) for m in data['epn']['metadata']]
    for field in ('scene_indices','future','future_mask','target_hist','ego_plan'):
        np.testing.assert_array_equal(d[field],data['epn'][field])
    # Import metric kernel from EPN, the evaluator that explicitly uses it.
    sys.path.insert(0,str(ROOT/'EPN'))
    from utils import maskedMSETest
    # PFL and EPN dataset sources implement the identical target/coordinate contract.
    sys.path.insert(0,str(ROOT/'PFL'))
    preds={};metrics={};keep=None
    for model in data:
        k,p,r=estimates(data[model]);preds[model]=p;metrics[model]=r
        if keep is not None: np.testing.assert_array_equal(keep,k)
        keep=k
    selected,audit,selection=select(d,keep,metrics,args.selection)
    ds=dataset()
    assert (ds.hist_len,ds.fut_len,ds.t_h,ds.t_f,ds.d_s)==(16,25,30,50,2)
    np.testing.assert_array_equal(d['scene_indices'],np.unique(np.linspace(0,len(ds)-1,512,dtype=np.int64)))
    # Validate the actual manifest membership and evaluator target ordering for ALL
    # cached instances, not just the two winners of the selection rule.
    manifest=target_metadata(ds,d['scene_indices'].tolist(),'ngsim',30.)
    assert [key(m) for m in manifest]==[key(m) for m in d['metadata']]
    write_csv(OUT/'fig02_selection_audit.csv',audit)
    rows=[]
    for category,i in selected:
        row=dict(audit[i],interaction_type=category,
                 category_count=selection['categories'][category]['count'],
                 category_median_delta_rmse_m=selection['categories'][category]['median_epn_minus_pfl_medoid_rmse_m'],
                 selection_reason=selection['selection_reason'],
                 population='prediction-blind 512-scene grid over NGSIM test; not full test',
                 interaction_threshold=selection['interaction_75_percentile_threshold'])
        rows.append(row)
    write_csv(OUT/'fig02_selected_scenes.csv',rows)
    print(json.dumps(rows,indent=2),flush=True)
    geo=draw(ds,d,keep,preds,metrics,selected,audit)
    # Release the full test dataset before subprocess replay, retaining all masks
    # and original batch composition in each independently imported model process.
    del ds
    if not args.skip_inference_verification:
        old=[str(r['cache_index']) for r in rows]
        for model in ('pfl','epn'):
            subprocess.run([sys.executable,__file__,'--verify-model',model,'--indices',*old],check=True,cwd=ROOT)
    after={m:digest(p) for m,p in CKPT.items()};assert before==after
    selection_text = (
        'Scenes were selected post hoc to illustrate PFL advantages: within each interaction '
        'category in the top interaction-strength quartile, the selected scene maximizes the '
        'smaller of the EPN-minus-PFL mean and medoid trajectory-RMSE gains, requiring both '
        'gains to be positive (seed 0). These favourable examples are not representative of '
        'the test distribution and do not establish overall model superiority. '
        if args.selection == 'pfl_advantage' else
        'Within the top interaction-strength quartile, each scene is closest to its category '
        'median EPN-minus-PFL medoid trajectory RMSE, without a model-win filter (seed 0). ')
    caption=(f'Fig. 2. PFL two-stage and EPN predictions for longitudinal (top) and lateral '
        '(bottom) interactions in NGSIM. Each panel shows K=6 candidates (the first six of K=20 '
        'evaluator draws), 3 s histories (16 states), and 5 s futures (25 states). Sample mean '
        'and medoid are GT-independent; the endpoint-oracle candidate uses the GT endpoint '
        'for analysis only. Paired panels share target-centred coordinates, axis limits, and '
        'aspect ratios, with +s to the right and the lateral axis expanded. Selection is restricted '
        f'to {int(keep.sum()):,} common full-horizon instances from a prediction-blind 512-scene '
        'uniform grid over the ordered test split, not the complete test set. '
        + selection_text + 'Interaction strength is the inverse minimum ego-plan/target-CV distance. '
        'Annotations give 25-state trajectory RMSE in metres. '
        'Lane boundaries are approximate reconstructions from local lane-ID centre medians; '
        'vehicle glyphs retain recorded length-to-width ratios and motion-derived headings but are not to scale. Ego plans are the '
        'evaluator\'s quintic fits of recorded futures.')
    (OUT/'fig02_caption.txt').write_text(caption+'\n')
    report=dict(seed=SEED,K=6,generated_K=20,observation_states=16,prediction_states=25,
        population='cached 512-scene prediction-blind test grid, NOT full-test top quartile',
        total_test_scenes=1320103,cached_target_instances=len(keep),full_horizon_instances=int(keep.sum()),
        selection=selection,scene_keys=[list(key(r)) for r in rows],geometry=geo,
        metric='sqrt(sum_t(dx^2+dy^2)/25), EPN.utils.maskedMSETest; differs from endpoint-only 5s RMSE',
        checkpoint_sha256=before,checkpoints={m:str(p) for m,p in CKPT.items()},
        cache_protocols={m:x['protocol'] for m,x in data.items()},
        source_sha256={p:digest(ROOT/p) for p in ['experiments/common.py','experiments/scripts/evaluate_model.py',
            'experiments/scripts/collect_qualitative.py','PFL/data.py','EPN/data.py','EPN/utils.py','PFL/model.py','EPN/model.py']},
        test_manifest=str(dataset_paths('ngsim')['test'].resolve()),
        test_manifest_sha256=digest(dataset_paths('ngsim')['test']),
        verification='existing cache full batches replayed from verified checkpoints' if not args.skip_inference_verification else 'see existing model verification JSONs',
        files=['fig02_pfl_epn_qualitative.'+s for s in ('pdf','svg','png')])
    atomic_json(OUT/'fig02_provenance.json',report)
    subprocess.run(['pdftoppm','-png','-scale-to','1800','-singlefile',
        str(OUT/'fig02_pfl_epn_qualitative.pdf'),str(OUT/'fig02_pdf_render')],check=True)
    print('Completed: figures/fig02_pfl_epn_qualitative.{pdf,svg,png}',flush=True)


if __name__=='__main__':
    main()
