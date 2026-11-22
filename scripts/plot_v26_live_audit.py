#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication-ready English-labelled benchmark figures from real saved data."""
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'REPORTS/live_v26_audit_2026_11_22'
P=O/'charts';P.mkdir(parents=True,exist_ok=True)
f=O/'frozen/summary.json'
if not f.exists():raise SystemExit('Need frozen summary')
s=json.loads(f.read_text(encoding='utf-8'))
models=['v24_1_r4','auto','v24_1_r3_1','v24_1_r2_1_6','v24','v23','v25','vi']
names=['V26.1 SOTA','Auto Router','R3.1','R2.1.6','V24','V23','V25','Vietnamese*']
values=[s['models'][m]['micro_cer_pct'] for m in models]
fig,ax=plt.subplots(figsize=(9,5))
bars=ax.barh(names[::-1],values[::-1])
for b,v in zip(bars,values[::-1]):ax.text(b.get_width()+0.5,b.get_y()+b.get_height()/2,f'{v:.2f}%',va='center',fontsize=9)
ax.set_xlim(0,max(values)+15)
ax.set_xlabel('Micro CER, % (lower is better)')
ax.set_title('Cham OCR Production — Same 29 Cham Pages, 22 Nov 2026')
ax.text(.01,-.13,'*Vietnamese-only recognizer is outside its intended language on Cham pages',transform=ax.transAxes,fontsize=8)
fig.tight_layout();fig.savefig(P/'frozen_cer_by_model.png',dpi=180,bbox_inches='tight');plt.close(fig)
tiers=['1','2','3','4','5','6']
names_t=['Clean','Aged / noisy','Narrow gaps','Wavy baseline','Overlapping','Real camera']
fig,ax=plt.subplots(figsize=(9,5))
for m,label in [('v24_1_r4','V26.1'),('v25','V25'),('auto','Auto'),('v24_1_r3_1','R3.1')]:
 y=[s['by_tier'][m][t]['micro_cer_pct'] for t in tiers]
 ax.plot(names_t,y,marker='o',label=label)
ax.set_ylabel('CER, %')
ax.set_title('Error Increase by Difficulty — Paired E2E Test')
ax.tick_params(axis='x',rotation=20)
ax.legend(loc='upper left')
fig.tight_layout();fig.savefig(P/'frozen_cer_by_difficulty.png',dpi=180,bbox_inches='tight');plt.close(fig)
# Crops; dynamic as long as data exists
g=O/'crops/summary.json'
if g.exists():
 j=json.loads(g.read_text(encoding='utf-8'));codes=[m for m in models if m in j]
 counts=[sum(z.get('correct',0) for z in j[m]['symbols'].values()) for m in codes]
 totals=[sum(z.get('n',0) for z in j[m]['symbols'].values()) for m in codes]
 pts=[100*x/max(1,n) for x,n in zip(counts,totals)]
 fig,ax=plt.subplots(figsize=(9,5))
 b=ax.barh([dict(zip(models,names))[m] for m in codes][::-1],pts[::-1])
 for bar,x,n in zip(b,counts[::-1],totals[::-1]):
  ax.text(bar.get_width()+.5,bar.get_y()+bar.get_height()/2,f'{x}/{n}',va='center',fontsize=9)
 ax.set_xlim(0,116);ax.set_xlabel('Aligned symbol hit rate (%)')
 ax.set_title('14 Punctuation Signs — Recognition-only Test (partial until 3/group)')
 fig.tight_layout();fig.savefig(P/'symbols_aligned_accuracy.png',dpi=180,bbox_inches='tight');plt.close(fig)
# Stress after source data
h=O/'stress/summary.json'
if h.exists():
 j=json.loads(h.read_text(encoding='utf-8'))
 labs=sorted(set(j.get('by_tier',{}).get('v24_1_r4',{})))
 labs=[l for l in labs if l.startswith('stress_')]
 if labs:
  fig,ax=plt.subplots(figsize=(11,5))
  for m,label in [('v24_1_r4','V26.1'),('v25','V25'),('auto','Auto'),('v24_1_r3_1','R3.1')]:
   y=[j['by_tier'].get(m,{}).get(k,{}).get('micro_cer_pct') for k in labs]
   ax.plot([s.replace('stress_','') for s in labs],y,marker='o',label=label)
  ax.tick_params(axis='x',rotation=40)
  ax.set_ylabel('CER, %');ax.set_title('Image Quality Transformations: E2E CER')
  ax.legend();fig.tight_layout()
  fig.savefig(P/'stress_by_transform.png',dpi=180,bbox_inches='tight');plt.close(fig)
print('PLOTS',[(x.name,x.stat().st_size) for x in P.glob('*.png')])
