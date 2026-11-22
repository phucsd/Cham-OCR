from pathlib import Path
from PIL import Image
import json
ROOT=Path(__file__).resolve().parents[1]
CASES=json.loads((ROOT/'REPORTS/live_quality_2026_11_20/frozen_test_manifest.json').read_text(encoding='utf-8'))['cases']
item=[next(c for c in CASES if c['tier']==x) for x in [1,2]]
a=[Image.open(ROOT/c['path']).convert('RGB') for c in item]
out=ROOT/'REPORTS/live_v26_audit_2026_11_22/ui/two_page_sample.pdf'
out.parent.mkdir(parents=True,exist_ok=True)
a[0].save(out,'PDF',save_all=True,append_images=a[1:],resolution=150)
print('CREATED_TEST_PDF',out,'bytes',out.stat().st_size,'pages',len(a))
