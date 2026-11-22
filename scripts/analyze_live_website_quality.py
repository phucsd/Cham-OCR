#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Error attribution and paired-model comparison for live Cham OCR benchmark."""
import json, collections, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'REPORTS/live_quality_2026_11_20'
DATA=OUT/'results.jsonl'

def align(g,p):
    # Minimum-edit alignment; ties prefer substitution to preserve useful confusion pairs.
    m,n=len(g),len(p)
    mat=[[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): mat[i][0]=i
    for j in range(n+1): mat[0][j]=j
    for i in range(1,m+1):
        for j in range(1,n+1):
            mat[i][j]=min(mat[i-1][j]+1,mat[i][j-1]+1,
                          mat[i-1][j-1]+(g[i-1]!=p[j-1]))
    i,j=m,n; ops=[]
    while i or j:
        if i and j and mat[i][j]==mat[i-1][j-1]+(g[i-1]!=p[j-1]):
            if g[i-1]!=p[j-1]:ops.append(('S',g[i-1],p[j-1]))
            i-=1;j-=1
        elif i and mat[i][j]==mat[i-1][j]+1:
            ops.append(('D',g[i-1],'∅'));i-=1
        else:
            ops.append(('I','∅',p[j-1]));j-=1
    return ops[::-1]

def category(c):
    if c in (' ','\n','\t'):return 'Spacing / linebreak'
    if c=='∅':return 'Insertion/deletion'
    if not c:return 'Other'
    x=ord(c)
    if 0xAA50<=x<=0xAA59:return 'Cham digit'
    if 0xAA5D<=x<=0xAA5F:return 'Cham punctuation'
    if 0xAA29<=x<=0xAA32:return 'Cham vowel / diacritic'
    if 0xAA33<=x<=0xAA36:return 'Cham medial'
    if 0xAA40<=x<=0xAA4D:return 'Cham coda'
    if 0xAA00<=x<=0xAA28:return 'Cham base consonant / vowel'
    if c.isdigit():return 'Latin digit'
    if c.isascii() and not c.isalnum():return 'ASCII punctuation'
    return 'Other'

def main():
    rows=[json.loads(s) for s in DATA.read_text(encoding='utf-8').splitlines() if s.strip()]
    models=list(dict.fromkeys(r['model'] for r in rows))
    good=[r for r in rows if not r.get('error') and r.get('actual_model')==r['model']]
    lines=['# Cham OCR Live: Detailed Failure Audit','',
           'All results newly sampled at production URL; see REPORT.md for methods.','',
           '## Categorized character errors','']
    data={}
    for model in models:
        rs=[r for r in good if r['model']==model and r['tier']!=0]
        if not rs:continue
        opcounts=collections.Counter();confusions=collections.Counter()
        charclass=collections.Counter(); all_errors=[]
        for r in rs:
            g='\n'.join(r['gt_lines']);p='\n'.join(r['pred_lines'])
            ops=align(g,p)
            for typ,src,dst in ops:
                opcounts[typ]+=1;charclass[category(src if src!='∅' else dst)]+=1
                confusions[(typ,src,dst)]+=1
            all_errors.append(dict(id=r['id'],tier=r['tier'],cer=r['cer'],
                edits=r['edit_chars'],gt_chars=r['gt_chars'],gt_count=r['gt_count'],
                pred_count=r['pred_count'],gt_lines=r['gt_lines'],pred_lines=r['pred_lines']))
        ranked=sorted(all_errors,key=lambda x:x['cer'],reverse=True)
        lines+=['### '+model,'',
            'Total edits: %d (substitutions %d / deletions %d / insertions %d)'%(
                sum(opcounts.values()),opcounts['S'],opcounts['D'],opcounts['I']),'',
            'By source-character category: '+', '.join(k+': '+str(v) for k,v in charclass.most_common()),'',
            'Top confusions:','',
            '| Operation | GT | OCR | Count |','|---|---|---|---:|']
        for (op,src,dst),cnt in confusions.most_common(15):
            esc=lambda c: 'SPACE' if c==' ' else 'NEWLINE' if c=='\n' else c
            lines.append('| %s | %s | %s | %s |'%(op,esc(src),esc(dst),cnt))
        lines+=['','Worst three documents:','']
        for r in ranked[:3]:
            lines.append('- %s (tier %d), CER %.2f%%, edits %d/%d, lines %d/%d'%(
                r['id'],r['tier'],r['cer']*100,r['edits'],r['gt_chars'],
                r['pred_count'],r['gt_count']))
            lines.append('  GT first line: '+str(r['gt_lines'][:1]))
            lines.append('  OCR first line: '+str(r['pred_lines'][:1]))
        lines.append('')
        data[model]=dict(operations=opcounts,character_categories=charclass,
                         top_confusions=[dict(op=k[0],gt=k[1],ocr=k[2],count=v)
                                         for k,v in confusions.most_common(30)],
                         worst=ranked[:10])
    lines+=['## Paired comparison on identical inputs','',
            '| Newer model | Comparator | Same images | Newer wins | Ties | Newer loses | Mean CER gain (points) |',
            '|---|---|---:|---:|---:|---:|---:|']
    paired={}
    ref='v24_1_r4'
    m={(r['id'],r['model']):r for r in good if r['tier']!=0}
    for other in models:
        if other==ref:continue
        ids=[r['id'] for r in good if r['model']==ref and r['tier']!=0 and (r['id'],other) in m]
        diffs=[100*(m[(i,other)]['cer']-m[(i,ref)]['cer']) for i in ids]
        win=sum(v>1e-9 for v in diffs);tie=sum(abs(v)<=1e-9 for v in diffs);loss=sum(v < -1e-9 for v in diffs)
        mean=round(statistics.mean(diffs),2) if diffs else None
        lines.append('| %s | %s | %d | %d | %d | %d | %s |'%(
            ref,other,len(ids),win,tie,loss,mean if mean is not None else '—'))
        paired[other]=dict(compared=len(ids),r4_wins=win,ties=tie,r4_loses=loss,
                           mean_cer_gain_points=mean,
                           by_image=[dict(id=i,cer_r4=m[(i,ref)]['cer'],cer_other=m[(i,other)]['cer'])
                                     for i in ids])
    (OUT/'ERROR_DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'error_diagnostics.json').write_text(json.dumps(dict(models=data,paired=paired),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Diagnostic complete:',len(good),'valid live results')

if __name__=='__main__':main()
