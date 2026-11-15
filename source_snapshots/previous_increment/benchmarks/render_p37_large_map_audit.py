"""Render representative cached failures and their reference-route profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import map_coordinates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run=args.run_dir
    auditdir=run/'diagnostics/large_map_failure_20261115'
    controlsdir=run/'diagnostics/large_map_decoder_controls_20261115'
    report=json.loads((auditdir/'report.json').read_text());byid={r['task_id']:r for r in report['tasks']}
    controls=json.loads((controlsdir/'report.json').read_text());control_byid={r['task_id']:r for r in controls['tasks']}
    original=json.loads((run/'evaluation/paired_best_val_epoch15/report.json').read_text());origbyid={r['task_id']:r for r in original['tasks']}
    with np.load(original['manifest']['binding']['data_path'],allow_pickle=True) as d:
        maps,mapids,starts,goals,routes=d['unique_maps'],d['map_ids'],d['starts'],d['goals'],d['paths']
    cases=[(82444,1,'Early progress gap'),(58875,0,'Progress mismatch leaves the route incomplete'),(11449,1,'Remaining route-field error')]
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,2,figsize=(12,14),gridspec_kw={'width_ratios':[1.25,1]})
    for row,(tid,c,title) in enumerate(cases):
        task=byid[tid];image=maps[int(mapids[tid])];ref=routes[tid]
        with np.load(run/'evaluation/paired_best_val_epoch15'/origbyid[tid]['field_cache'],allow_pickle=False) as d:field=d['fields'][0,c]
        with np.load(auditdir/'paths'/f'task_{tid:06d}.npz',allow_pickle=False) as d:
            baseline=d['prediction'][c]
            green=d['predicted_occupancy_oracle_progress_extended'][c] if tid!=11449 else d['oracle_both'][0]
        with np.load(controlsdir/'paths'/f'task_{tid:06d}.npz',allow_pickle=False) as d:smooth=d['progress_support_weighted_smooth7'][c]
        ax=axes[row,0]
        ax.imshow(image,cmap='Greys',vmin=0,vmax=1,origin='upper',alpha=.65)
        overlay=np.ma.masked_where(field[0]<=.1,field[0].clip(0,1))
        ax.imshow(overlay,cmap='Blues',vmin=0,vmax=1,origin='upper',alpha=.6)
        ax.plot(green[:,0],green[:,1],color='#24945d',lw=1.7,alpha=.9)
        if tid==58875:ax.plot(smooth[:,0],smooth[:,1],color='#7b3fb3',lw=1.7,ls='--')
        ax.plot(baseline[:-1,0],baseline[:-1,1],color='#d22f3a',lw=2.1)
        ax.plot(baseline[-2:,0],baseline[-2:,1],color='#f39b23',lw=2,ls='--')
        ax.scatter(*starts[tid],c='#24945d',s=55,marker='o',edgecolors='white',zorder=9)
        ax.scatter(*goals[tid],c='black',s=65,marker='*',edgecolors='white',zorder=9)
        lastjump=task['baseline_collisions'][c]['final_step_px']
        ax.set_title(f'{title}\nTask {tid}, {task["shape"]}, candidate {c}\nForced final link: {lastjump:.0f} px',loc='left',fontsize=11)
        ax.set_xlabel('x (grid pixels)');ax.set_ylabel('y (grid pixels)')
        lens=np.linalg.norm(np.diff(ref.astype(float),axis=0),axis=1);cum=np.r_[0,lens.cumsum()]
        arc=np.linspace(0,cum[-1],max(128,int(cum[-1]*2)));q=arc/cum[-1]
        xy=np.stack([np.interp(arc,cum,ref[:,k]) for k in (0,1)],axis=1)
        occ=map_coordinates(field[0],[xy[:,1],xy[:,0]],order=1,mode='nearest').clip(0,1)
        prog=map_coordinates(field[1],[xy[:,1],xy[:,0]],order=1,mode='nearest').clip(0,1)
        ax=axes[row,1]
        ax.plot(q,occ,color='#999999',lw=1,alpha=.8,label='Predicted occupancy on reference')
        ax.plot(q,prog,color='#2371bd',lw=1.3,label='Predicted progress on reference')
        ax.plot(q,q,color='black',ls='--',lw=1,label='Reference arc-length progress')
        ax.axhline(.1,color='#aaa',ls=':',lw=1)
        ax.fill_between(q,0,1,where=occ<.1,color='#eea13f',alpha=.2,label='Reference support below threshold')
        ax.set(xlim=(0,1),ylim=(-.025,1.05),xlabel='Fraction of reference route length',ylabel='Field value')
        ax.grid(alpha=.15)
        first=task['variants']['prediction']['diagnostics']['first_missing_support_step'][c]
        info=f'First missing-support step: {first}/512' if first<512 else 'No support gap; final link still invalid'
        recall=task['predicted_field_metrics'][c]['teacher_corridor_occupancy_recall_at_point1']
        ax.set_title(f'{info}\nReference corridor support recall: {recall:.1%}',loc='left')
        if row==0:ax.legend(fontsize=8,loc='lower right')
    handles=[Line2D([0],[0],color='#d22f3a',lw=2,label='Baseline prefix'),
             Line2D([0],[0],color='#f39b23',lw=2,ls='--',label='Forced final link (rejected)'),
             Line2D([0],[0],color='#24945d',lw=2,label='Teacher-assisted diagnostic path'),
             Line2D([0],[0],color='#7b3fb3',lw=2,ls='--',label='Predicted-progress smoothing (middle case)')]
    fig.legend(handles=handles,loc='lower center',ncol=2,fontsize=10,frameon=False,bbox_to_anchor=(.5,.012))
    fig.suptitle('Frozen epoch 15: route support is often present, but progress and local decoding disagree',fontsize=14,y=.985)
    fig.text(.5,.958,'Teacher-assisted paths use reference labels for diagnosis and are not model performance.',ha='center',fontsize=10,color='#555')
    fig.subplots_adjust(top=.91,bottom=.085,hspace=.50,wspace=.28)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=150)
    plt.close(fig)
    print(args.output)


if __name__=='__main__':main()
