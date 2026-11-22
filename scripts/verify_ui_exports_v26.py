import json,zipfile,xml.etree.ElementTree as ET,sys
from pathlib import Path
out=Path(__file__).resolve().parents[1]/'REPORTS/live_v26_audit_2026_11_22/ui'
f={name:next(out.glob('e2e_'+name+'_*')) for name in ['txt','md','json','xml','zip']}
txt=f['txt'].read_text(encoding='utf-8-sig')
md=f['md'].read_text(encoding='utf-8-sig')
js=json.loads(f['json'].read_text(encoding='utf-8-sig'))
root=ET.parse(f['xml']).getroot()
lines=[x for x in root.iter() if x.tag.split('}')[-1]=='TextLine']
with zipfile.ZipFile(f['zip']) as z:
   names=z.namelist()
   zipresult=dict(files=len(names),names=names[:30],crc_error=z.testzip(),
       unsafe_paths=[n for n in names if n.startswith('/') or '..' in Path(n).parts])
result={'txt':dict(length=len(txt),Cham_characters=sum(0xAA00<=ord(c)<=0xAA5F for c in txt),lines=len(txt.splitlines())),
  'md':dict(length=len(md),has_metadata='Confidence' in md or 'confidence' in md,has_unicode=any(0xAA00<=ord(c)<=0xAA5F for c in md)),
  'json':dict(keys=list(js) if isinstance(js,dict) else 'array',length=len(f['json'].read_bytes())),
  'xml':dict(root_tag=root.tag,recognized_TextLine_elements=len(lines)),
  'zip':zipresult}
(out/'export_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True,indent=2))
