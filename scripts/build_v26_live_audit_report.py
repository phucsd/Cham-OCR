#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build real-production Vietnamese results and paired statistical comparison."""
import csv,json,collections,random,statistics,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
O=R/'REPORTS/live_v26_audit_2026_11_22'
sys.path.insert(0,str(R/'scripts'))
from benchmark_all_live_v26 import MODELS
from analyze_live_website_quality import align,category
S=['frozen','geometry','stress','vietnamese']
N={'v24_1_r4':'V26.1 SOTA','v25':'V25 SOTA','auto':'Auto Router','v24_1_r3_1':'R3.1','v24_1_r2_1_6':'R2.1.6','v24':'V24 Baseline','v23':'V23 Legacy','vi':'Vietnamese'}
def load(p):
 if not p.exists():return []
 r={}
 for l in p.read_text(encoding='utf-8').splitlines():
  try:
   a=json.loads(l);r[(a['id'],a['model'])]=a
  except Exception: pass
 return list(r.values())
D={s:load(O/s/'results.jsonl') for s in S}
def valid(r):return not r.get('error') and r.get('actual_model')==r.get('model') and isinstance(r.get('edit_chars'),int)
def neg(r):return str(r.get('tier','')).lower() in ('0','negative')
def agg(items):
 a=[r for r in items if valid(r) and not neg(r)]
 if not a:return dict(n=0,attempts=len(items),failures=len(items),cer=None,wer=None,lines=None,exact=None,backend=None,wall=None,p95=None)
 chars=sum(r['gt_chars'] for r in a);words=sum(r['gt_words'] for r in a)
 timings=sorted(r['wall_sec'] for r in a if isinstance(r.get('wall_sec'),(int,float)))
 back=[float(r.get('backend_sec') or 0) for r in a]
 return dict(n=len(a),attempts=len(items),failures=sum(not valid(r) for r in items),
  cer=round(100*sum(r['edit_chars'] for r in a)/max(1,chars),2),
  wer=round(100*sum(r['edit_words'] for r in a)/max(1,words),2),
  lines=round(100*sum(r['gt_count']==r['pred_count'] for r in a)/len(a),2),
  exact=round(100*sum(r['exact'] for r in a)/len(a),2),
  backend=round(statistics.median(back),2),wall=round(statistics.median(timings),2) if timings else None,
  p95=round(timings[min(len(timings)-1,int(.95*(len(timings)-1)))],2) if timings else None)
def fmt(x):return '-' if x is None else str(x)
def table(h,rows):
 return ['| '+' | '.join(map(str,h))+' |','|'+'|'.join(['---']+['---:']*(len(h)-1))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows]
def pair(rs,left,right):
 a={r['id']:r for r in rs if r['model']==left and valid(r) and not neg(r)}
 b={r['id']:r for r in rs if r['model']==right and valid(r) and not neg(r)}
 items=[(a[k],b[k]) for k in sorted(a.keys()&b.keys()) if a[k].get('input_sha256')==b[k].get('input_sha256')]
 if not items:return None
 diffs=[100*(y['cer']-x['cer']) for x,y in items]
 wins=sum(x>1e-6 for x in diffs);ties=sum(abs(x)<=1e-6 for x in diffs)
 rng=random.Random(20261122);sample=[]
 for i in range(2500):
  s=[items[rng.randrange(len(items))] for _ in items]
  sample.append(100*(sum(y['edit_chars']-x['edit_chars'] for x,y in s))/max(1,sum(x['gt_chars'] for x,y in s)))
 sample.sort()
 return dict(n=len(items),wins=wins,ties=ties,losses=len(items)-wins-ties,
  micro_delta=round(100*sum(y['edit_chars']-x['edit_chars'] for x,y in items)/max(1,sum(x['gt_chars'] for x,y in items)),2),
  ci95=[round(sample[62],2),round(sample[2436],2)])
def main():
 lines=['# BÁO CÁO KIỂM THỬ TOÀN DIỆN OCR.CHAM.ASIA — 22/11/2026','',
 '## 1. Phạm vi và quy tắc','',
 'Đo E2E trên website production bằng POST /ocr; cùng ảnh/nhãn GT, DBNet hybrid và đọc rows. **Cả 8 model/chế độ** được đo trên trọn bộ 30 ảnh cố định. Các nhóm mở rộng có chọn mẫu đại diện và ưu tiên những model chính; số ảnh/model được ghi rõ từng bảng để tránh ngộ nhận là toàn bộ 8 model đã chạy mọi biến thể. Vietnamese là model khác ngôn ngữ, Auto là router không phải checkpoint độc lập.','',
 'Micro CER = tổng Levenshtein edits ký tự/tổng ký tự GT trên trang có chữ, Unicode NFC. WER = whitespace-token, chỉ phụ trợ đối với chữ Chăm. Số dòng khớp không chứng minh polygon IoU/precision/recall.','',
 'Ảnh test được lưu có SHA256, response có fingerprint checkpoint. Synthetic/biến đổi ảnh có tương quan với ảnh gốc và chưa chứng minh độc lập dữ liệu train. Mẫu real-camera giới hạn, nên không có kết luận tuyệt đối cho tất cả sách/tài liệu thực tế.','',
 'Độ trễ được đo trong điều kiện nhiều benchmark đồng thời trên cùng production Space: chỉ có giá trị quan sát thực tế, không phải phép đo tốc độ model trong môi trường cô lập. Muốn so inference tốc độ chuẩn cần chạy từng model tuần tự sau warm-up với concurrency cố định.','',
 '## 2. Khối lượng chạy thật','']
 summary={};allrows=[];comparisons={}
 counts=[]
 for s in S:
  rows=D[s];u=len({r['id'] for r in rows});num=len(rows)
  counts.append([s,u,num,sum(not valid(r) for r in rows)])
 lines+=table(['Suite','Ảnh riêng đã có kết quả','POST /ocr đã ghi','Response lỗi/mismatch'],counts)
 lines+=['','Các suite có số ảnh ít hơn thiết kế là chưa hoàn thành, không được diễn giải là PASS.','',
         '## 3. Bảng xếp hạng theo 8 model và bộ dữ liệu','']
 for s in S:
  lines+=['','### '+s,'']
  summary[s]={};mat=[]
  for m in MODELS:
   a=agg([r for r in D[s] if r['model']==m])
   summary[s][m]=a
   mat.append([N[m],a['n'],fmt(a['cer'])+'%',fmt(a['wer'])+'%',fmt(a['lines'])+'%',fmt(a['exact'])+'%',fmt(a['backend'])+'s',fmt(a['p95'])+'s',a['failures']])
   allrows.append([s,m,a['n'],a['attempts'],a['failures'],a['cer'],a['wer'],a['lines'],a['exact'],a['backend'],a['wall'],a['p95']])
  lines+=table(['Model','Trang chữ','CER','WER','Đúng số dòng','Đúng toàn trang','Backend P50','Wall P95','Lỗi API'],mat)
 lines+=['','**Ghi chú:** Không xếp hạng model tiếng Việt bằng CER chữ Chăm. Các trang song ngữ phải xem riêng theo ngôn ngữ, còn toàn-trang CER có thể bị chi phối bởi tỷ lệ Chăm/Việt.','',
 '## 4. CER từ dễ đến khó / điều kiện ảnh','']
 for s in S:
  tiers=sorted({str(x['tier']) for x in D[s] if not neg(x)})
  if not tiers:continue
  lines+=['','### '+s,'']
  mat=[]
  for t in tiers:
   vals=[]
   for m in MODELS:
    q=agg([r for r in D[s] if r['model']==m and str(r['tier'])==t])
    vals.append(fmt(q['cer'])+'% ('+str(q['n'])+')')
   mat.append([t]+vals)
  lines+=table(['Độ khó/biến thể']+[N[m] for m in MODELS],mat)
 lines+=['','## 5. Thống kê so sánh từng ảnh ghép cặp (trùng SHA256)','',
  'CER đối thủ − V26.1: dương nghĩa V26.1 tốt hơn. CI95 bootstrap mẫu theo ảnh 2.500 lần; ảnh biến dạng cùng nguồn không độc lập nên khoảng tin cậy có thể lạc quan.','']
 for s in S:
  mat=[]
  for m in MODELS[1:]:
   v=pair(D[s],'v24_1_r4',m)
   if not v:continue
   comparisons[s+'_'+m]=v
   mat.append([N[m],v['n'],v['wins'],v['ties'],v['losses'],v['micro_delta'],str(v['ci95'][0])+' → '+str(v['ci95'][1])])
  if mat:
   lines+=['','### '+s,'']
   lines+=table(['Đối chứng','Ảnh','V26 thắng','Hòa','V26 thua','Δ CER điểm %','95% CI'],mat)
 prev=load(R/'REPORTS/live_quality_2026_11_20/results.jsonl')
 old={r['id']:r for r in prev if r['model']=='v24_1_r4' and valid(r) and not neg(r)}
 new={r['id']:r for r in D['frozen'] if r['model']=='v24_1_r4' and valid(r) and not neg(r)}
 oldpairs=[(new[k],old[k]) for k in sorted(new.keys()&old.keys()) if new[k].get('input_sha256')==old[k].get('input_sha256')]
 if oldpairs:
  oldcer=100*sum(b['edit_chars'] for a,b in oldpairs)/max(1,sum(b['gt_chars'] for a,b in oldpairs))
  newcer=100*sum(a['edit_chars'] for a,b in oldpairs)/max(1,sum(a['gt_chars'] for a,b in oldpairs))
  lines+=['','## 6. Hồi quy phiên bản cũ 20/11 vs phiên bản mới 22/11','',
  'Trùng ảnh và SHA256: '+str(len(oldpairs))+' trang có chữ; R4 cũ fingerprint 85525b0ca091acaf CER '+f'{oldcer:.2f}%'+'; V26.1 fingerprint 1deee5f5ff351070 CER '+f'{newcer:.2f}%'+'; thay đổi '+f'{newcer-oldcer:+.2f}'+' điểm %.','',
  'Kết quả là E2E ở 2 ngày đo khác nhau, chịu ảnh hưởng của cả backend/detector/normalizer, không chứng minh nguyên nhân chỉ do thay checkpoint.']
 crop=O/'crops/summary.json'
 if crop.exists():
  c=json.loads(crop.read_text(encoding='utf-8'))
  lines+=['','## 7. Dấu câu 14 loại và dấu gạch nối 7 ngữ cảnh','']
  mat=[]
  for m in MODELS:
   v=c.get(m,{});sym=v.get('symbols',{});hy=v.get('hyphens',{})
   mat.append([N[m],v.get('cases',0),fmt(v.get('cer_pct'))+'%',fmt(v.get('full_line_exact_pct'))+'%',
       str(sum(x['correct'] for x in sym.values()))+'/'+str(sum(x['n'] for x in sym.values())),
       str(sum(x['count_correct'] for x in hy.values()))+'/'+str(sum(x['n'] for x in hy.values())),v.get('errors',0)])
  lines+=table(['Model','Crop','CER','Full line exact','Đúng symbol','Đúng số dấu -','API errors'],mat)
  lines+=['','Symbol được chấm bằng alignment Levenshtein tại vị trí GT; hyphen chỉ chấm số lượng, không xác nhận vị trí. Đây là crop đã được tách sẵn, không đo lỗi detector.','',
    '### Từng ký hiệu']
  keys=sorted(set(k for x in c.values() for k in x.get('symbols',{})))
  lines+=table(['Ký hiệu']+[N[m] for m in MODELS],[
   [('BACKSLASH' if k=='\\' else k)]+[
     str(c.get(m,{}).get('symbols',{}).get(k,{}).get('correct',0))+'/'+
     str(c.get(m,{}).get('symbols',{}).get(k,{}).get('n',0)) for m in MODELS] for k in keys])
 else:lines+=['','## 7. Dấu câu','Bộ crops đang chạy, chưa đủ kết quả tổng kết.']
 lines+=['','## 8. Top lỗi nặng nhất của model mặc định','']
 for s in S:
  a=sorted((r for r in D[s] if r['model']=='v24_1_r4' and valid(r) and not neg(r)),key=lambda x:x['cer'],reverse=True)[:10]
  if a:
   lines+=['','### '+s]
   lines+=table(['Ảnh','Độ khó','CER','Dòng OCR/GT','Edit'],[
    [r['id'],r['tier'],f"{r['cer']*100:.2f}%",str(r['pred_count'])+'/'+str(r['gt_count']),r['edit_chars']] for r in a])
 confusions=collections.Counter();cat=collections.Counter()
 for r in D['frozen']:
  if r['model']!='v24_1_r4' or not valid(r) or neg(r):continue
  g='\n'.join(r.get('gt_lines',[]));p='\n'.join(r.get('pred_lines',[]))
  for typ,src,dst in align(g,p):
   confusions[(typ,src,dst)]+=1;cat[category(src if src!='∅' else dst)]+=1
 lines+=['','## 9. Lỗi ký tự trên bộ cố định','',
  'Nhóm: '+', '.join(str(k)+'='+str(v) for k,v in cat.most_common()),'']
 def lab(c):return 'SPACE' if c==' ' else 'NEWLINE' if c=='\n' else c
 lines+=table(['Operation','GT','OCR','Số lượt'],[[tp,lab(src),lab(dst),n] for (tp,src,dst),n in confusions.most_common(22)])
 lines+=['','## 10. Ảnh không chữ / false positives','']
 lines+=table(['Model','Ảnh âm tính','Đúng (0 dòng)','Nhận nhầm (hơn 0)','API lỗi'],[
  [N[m],len(a),sum(valid(r) and r.get('pred_count')==0 for r in a),
   sum(valid(r) and r.get('pred_count',0)>0 for r in a),sum(not valid(r) for r in a)]
  for m in MODELS for a in [[r for s in ['frozen','stress'] for r in D[s] if r['model']==m and neg(r)]]])
 lines+=['','## 11. Kiểm thử Chrome, nhập PDF và xuất định dạng','']
 x=O/'ui/exports_pdf_results.json'
 if x.exists():
  v=json.loads(x.read_text(encoding='utf-8'))
  lines+=['','Playwright Chrome thực: '+', '.join(r['name']+' '+('PASS' if r.get('ok') else 'FAIL') for r in v['results'])+'.',
  'Browser JS errors: '+str(len(v['pageErrors']))+'; failed browser requests: '+str(len(v['failedRequests']))+'.']
 x=O/'ui/export_validation.json'
 if x.exists():
  v=json.loads(x.read_text(encoding='utf-8'))
  lines+=['','Sau khi export: TXT gồm '+str(v['txt']['lines'])+' dòng, '+str(v['txt']['Cham_characters'])+' ký tự Chăm; PAGE-XML hợp lệ có '+str(v['xml']['recognized_TextLine_elements'])+' TextLine; ZIP gồm '+str(v['zip']['files'])+' entries, kiểm CRC='+str(v['zip']['crc_error'])+'.']
 x=O/'api_contract/results.json'
 if x.exists():
  v=json.loads(x.read_text(encoding='utf-8'))
  lines+=['','## 12. Kiểm thử HTTP và dữ liệu sai','']
  lines+=table(['Trường hợp','HTTP','Giây'],[[r['case'],r['http'],r['latency_s']] for r in v])
 x=O/'api_contract/invalid_model_fallback.json'
 if x.exists():
  v=json.loads(x.read_text(encoding='utf-8'));z=v['recognizers'][0]
  lines+=['','**P0/P1 phát hiện thật:** Model không tồn tại '+v['requested']+' vẫn HTTP '+str(v['http'])+'; metadata nhận tên model được yêu cầu, nhưng model_dir='+z['model_dir']+', SHA='+z['weights_sha256_16']+' chính là V23 Legacy. Nguy cơ benchmark/model provenance giả.','']
 # Character-level failure diagnosis and confidence calibration.
 x=O/'script_leakage_analysis.json'
 if x.exists():
  v=json.loads(x.read_text(encoding='utf-8'))
  lines+=['','## 13. Dự đoán nhầm chữ Latin trên văn bản Chăm','',
   'Nhãn chuẩn 29 trang hoàn toàn không chứa chữ cái Latin. Đếm theo Unicode Name LATIN của ký tự đầu ra OCR, không tính chữ số/ký hiệu.','']
  lines+=table(['Model','Trang','Chữ Latin do OCR tạo','Tỷ lệ Latin trong text OCR','Trang xuất hiện Latin'],[
   [N[m],v[m]['n'],v[m]['ocr_latin_letters'],str(v[m]['ocr_latin_letter_pct'])+'%',v[m]['pages_with_latin_letters']]
   for m in MODELS if m in v])
 x=O/'charset_order_audit.json'
 if x.exists():
  v=json.loads(x.read_text(encoding='utf-8'))
  lines+=['','**Kiểm tra giả thuyết sai thứ tự từ điển trên bản export local:** '+', '.join(
   N.get(m,m)+': '+str(z['embedded_len'])+' token, thứ tự YAML so với file từ điển '+('TRÙNG' if z['identical_order'] else 'KHÁC')
   for m,z in v.items())+'. Việc trùng thứ tự ở bản local giúp loại trừ lỗi đơn giản này, nhưng chưa kiểm tra trực tiếp nội dung dictionary được máy chủ dùng khi deploy.']
 x=O/'confidence_error_audit.json'
 if x.exists():
  v=json.loads(x.read_text(encoding='utf-8'))
  lines+=['','## 14. Hiệu chỉnh confidence và rủi ro tin sai','',
    'Confidence trung bình theo trang là thống kê mô tả, không tương đương độ chính xác ký tự; các model có nhiệt độ hiệu chỉnh khác nhau.','']
  lines+=table(['Model','Confidence TB','Pearson (confidence,CER)','Trang conf >=90% nhưng CER >=20%'],[
   [N[m],v[m]['mean_confidence'],v[m]['pearson_confidence_vs_cer'],v[m]['n_confidence_ge_90_cer_ge_20']]
   for m in MODELS if m in v])
 x=O/'v25_vs_r4_repro_examples.json'
 if x.exists():
  v=json.loads(x.read_text(encoding='utf-8'))
  lines+=['','## 15. Ca lỗi có thể tái hiện bằng OCR crop','',
   'Ví dụ crop trực tiếp: không phụ thuộc DBNet tách dòng. Xem JSON song song đầy đủ trong v25_vs_r4_repro_examples.json.']
  rr=[]
  for r in v.get('crops',[])[:4]:
   fix=lambda s:s.replace('|','∣').replace('\n',' ↵ ')[:100]
   rr.append([r['id'],r['cer_r4'],r['cer_v25'],fix(r['gt']),fix(r['r4']),fix(r['v25'])])
  lines+=table(['ID','R4 CER%','V25 CER%','GT','R4','V25'],rr)
 x=O/'digits/summary.json'
 if x.exists():
  v=json.loads(x.read_text(encoding='utf-8'))
  lines+=['','## 16. Chữ số Chăm U+AA50..U+AA59 (nhận dạng crop)','',
   'Ba font Noto Sans Cham thuộc các kiểu nét Light/Medium/SemiBold, mỗi font chứa các chuỗi 10 số Chăm và mẫu ngày/năm.','']
  lines+=table(['Model','Crops','CER số Chăm','Đúng cả crop','API errors'],[
    [N[m],v[m]['n'],str(v[m]['groups'].get('Cham 0-9',{}).get('cer_pct'))+'%',
     str(v[m]['groups'].get('Cham 0-9',{}).get('exact_pct'))+'%',v[m]['errors']]
    for m in MODELS if m in v])
 x=O/'pipeline/summary.json'
 if x.exists():
  v=json.loads(x.read_text(encoding='utf-8'))
  lines+=['','## 17. Cấu hình phát hiện dòng và thứ tự đọc','',
   'Ba cấu hình kiểm thử độc lập: valley/rows, adaptive/rows, dbnet/auto trên ba trang. Đây là mean per-page CER cho thăm dò, không tương đương micro CER ở phần chính.','']
  lines+=table(['Method']+[N[m] for m in ['v24_1_r4','v25','auto','v24_1_r3_1']],[
   [method]+[str(v.get(method,{}).get(m,{}).get('mean_cer_pct'))+'%' for m in ['v24_1_r4','v25','auto','v24_1_r3_1']]
   for method in ['valley','adaptive','dbnet']])
 lines+=['','## 18. Biểu đồ trực quan','',
    '![CER theo model](charts/frozen_cer_by_model.png)','',
    '![CER theo độ khó](charts/frozen_cer_by_difficulty.png)','',
    '![Nhận đúng dấu câu](charts/symbols_aligned_accuracy.png)','',
    '![Ảnh suy giảm](charts/stress_by_transform.png)']
 lines+=['','## 19. Đánh giá và khuyến nghị ưu tiên','',
 '1. **P0/High — Model provenance**: validate model_id từ allowlist; không tồn tại phải HTTP 400; actual_model và fingerprint phải xuất từ checkpoint thực, không echo ID người dùng.',
 '2. **P1/High — API**: thiếu image trả HTTP502 đã tái hiện; thay bằng HTTP400 JSON. Invalid method/reading_order nhận HTTP200; input validation và thông báo fallback cần rõ ràng.',
 '3. **P1/High — V25**: kiểm tra sự khác nhau giữa validation CER trong quá trình train và E2E CER production: dữ liệu train/GT, thứ tự Unicode, dictionary/token, preprocessing, width W480, detector crop, nguy cơ domain shift.',
 '4. **P1 — Chỉ đổi model mặc định sau frozen real-world gate**: micro CER trên nhiều bản thảo thật độc lập, đánh giá theo nguồn tài liệu, có CI theo tài liệu. Hiện số trang gốc camera chưa đủ.',
 '5. **P1 — Diacritics & ký hiệu**: khai thác top confusion, hard-mine các dấu Chăm, phụ âm cuối, dấu nối, các tổ hợp Unicode cluster; giải thích lỗi theo dấu/cặp ký tự.',
 '6. **P1 — Line detection**: cần GT polygon và F1 IoU, kiểm tra cắt dòng, nối dòng, hai cột, đảo chiều đọc, cong giấy một góc.',
 '7. **P2 — Auto router/đa ngữ**: gán nhãn ngôn ngữ theo dòng, đo route accuracy và riêng CER từng chữ; không xếp hạng Vietnamese theo Cham-only CER. API Auto hiện không xuất model/ngôn ngữ được chọn cho mỗi dòng, nên nên bổ sung selected_recognizer và router confidence để audit.',
 '8. **P2 — Regression vận hành**: kiểm tra PDF 100 trang, memory usage, mobile Safari/Firefox, mạng mất kết nối, video/camera, file nặng, file MIME sai, bảo toàn sửa thủ công.',
 '9. **P2 — Bổ sung ảnh thật**: thu ≥100 trang từ nhiều nguồn, chụp điện thoại, chữ viết tay, scan giấy ngả vàng, dấu phủ mực, độ sáng/góc khác nhau, split hold-out theo tài liệu tránh leakage.',
 '','## 20. Giới hạn và tái lập','',
 'Các bộ ảnh mới thuộc dạng ảnh tạo từ nguồn cũ (biến dạng và ghép có GT xác định); không chứng minh đã bao phủ mọi tình huống thực tế. Không test trực tiếp trên iOS/Android hoặc đánh giá hình học polygon với nhãn người gán.',
 'Nguyên tắc: chỉ tính các request JSONL đã ghi; model trực tiếp phải trả actual version khớp tên; router được ghi như router, không coi là checkpoint mới.',
 'Dữ liệu: REPORTS/live_v26_audit_2026_11_22/{frozen,geometry,stress,vietnamese,crops,pipeline}/results.jsonl, summary.json; nguồn GT manifest có SHA.',
 'Script: scripts/benchmark_all_live_v26.py, benchmark_live_v26_crops.py, benchmark_live_v26_pipeline.py, generate_v26_audit_cases.py, ui_export_pdf_e2e_v26.js.',
 'Không thay đổi production, không train model, không git push.','']
 out={'generated_utc':datetime.now(timezone.utc).isoformat(),'suites':summary,'comparisons':comparisons,'confusions':[dict(type=t,gt=a,ocr=b,count=n) for (t,a,b),n in confusions.most_common(35)],
       'categories':dict(cat),'calls':{s:len(D[s]) for s in S}}
 (O/'aggregate_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
 with (O/'model_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['suite','model','n_text','calls','failures','micro_cer_pct','micro_wer_pct','line_count_match_pct','exact_pct','median_backend_s','median_wall_s','p95_wall_s'])
  w.writerows(allrows)
 (O/'FINAL_REPORT_VI.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('GENERATED FINAL_REPORT_VI.md','suite counts',{s:len(D[s]) for s in S},'model_metrics',summary['frozen']['v24_1_r4'],flush=True)
if __name__=='__main__':main()
