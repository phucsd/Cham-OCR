#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic stress + Vietnamese + mixed-script holdout-style QA images."""
from pathlib import Path
from PIL import Image,ImageFont,ImageDraw,ImageEnhance,ImageFilter
import hashlib,io,json,random,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ocr-benchmark/data/live_v26_2026_11_22'
IMAGE=OUT/'images'
IMAGE.mkdir(parents=True,exist_ok=True)
frozen=json.loads((ROOT/'REPORTS/live_quality_2026_11_20/frozen_test_manifest.json').read_text(encoding='utf-8'))['cases']
sources=[]
for tier in [1,4,5,6]:
    x=next(c for c in frozen if c['tier']==tier)
    sources.append(x)
cases=[]
def add(suite,id,im,gt,tier,origin,transform):
    if im.mode!='RGB': im=im.convert('RGB')
    rel='ocr-benchmark/data/live_v26_2026_11_22/images/'+id+'.png'
    target=ROOT/rel; im.save(target,optimize=True)
    cases.append(dict(suite=suite,id=id,image_relpath=rel,tier=tier,
        gt_lines=gt,origin=origin,transform=transform,width=im.width,height=im.height,
        image_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
def mutate(im,key,seed):
    r=random.Random(seed)
    if key=='jpeg_35':
        b=io.BytesIO(); im.save(b,'JPEG',quality=35,subsampling=2);return Image.open(io.BytesIO(b.getvalue())).convert('RGB')
    if key=='blur_1_4':return im.filter(ImageFilter.GaussianBlur(radius=1.4))
    if key=='low_contrast':return ImageEnhance.Contrast(im).enhance(0.5)
    if key=='dark':return ImageEnhance.Brightness(im).enhance(0.58)
    if key=='rotate_plus_8':return im.rotate(8,expand=True,fillcolor='white',resample=Image.Resampling.BICUBIC)
    if key=='rotate_minus_11':return im.rotate(-11,expand=True,fillcolor='white',resample=Image.Resampling.BICUBIC)
    if key=='downsample_45':
        w,h=im.size;return im.resize((max(1,int(w*.45)),max(1,int(h*.45))),Image.Resampling.BILINEAR).resize((w,h),Image.Resampling.BILINEAR)
    if key=='shadow':
        out=im.copy(); w,h=out.size; wash=Image.new('RGB',(w,h))
        px=wash.load()
        for x in range(w):
            f=int(80*(1-x/max(1,w-1)))
            for y in range(h):px[x,y]=(f,f,f)
        return Image.blend(out,wash,0.32)
    if key=='noise_1pct':
        im=im.copy();px=im.load();w,h=im.size
        for i in range(int(w*h*0.01)):
            x=r.randrange(w);y=r.randrange(h);v=0 if r.random()<0.6 else 255;px[x,y]=(v,v,v)
        return im
    raise ValueError(key)
keys=['jpeg_35','blur_1_4','low_contrast','dark','rotate_plus_8','rotate_minus_11','downsample_45','shadow','noise_1pct']
for x in sources:
    blob=(ROOT/x['path']).read_bytes()
    assert hashlib.sha256(blob).hexdigest()==x['image_sha256']
    im=Image.open(io.BytesIO(blob)).convert('RGB')
    for key in keys:
        add('stress','stress_'+x['id']+'_'+key,mutate(im,key,20261122),x['gt_lines'],
            'stress_'+key,x['id'],key)
# Negative examples with no Cham or Vietnamese text
add('stress','negative_white',Image.new('RGB',(850,500),'white'),[],'negative','none','blank')
n=Image.new('RGB',(850,500),'white');draw=ImageDraw.Draw(n);r=random.Random(20261122)
for i in range(400):
    x=r.randrange(n.width);y=r.randrange(n.height);draw.point((x,y),fill=(r.randrange(180,256),)*3)
add('stress','negative_speckles',n,[],'negative','none','noise_no_text')
n=Image.new('RGB',(850,500),'white');draw=ImageDraw.Draw(n)
for y in range(40,490,43):draw.line((45,y,790,y),fill=(220,220,220),width=1)
add('stress','negative_ruled_page',n,[],'negative','none','lines_no_text')
# Vietnamese, diacritics, numbers, Latin symbols.
fontfiles=['arial.ttf','times.ttf','calibri.ttf']
texts=[
 ['Cộng hòa xã hội chủ nghĩa Việt Nam.','Đây là bản phiên âm tiếng Chăm và tiếng Việt.','Số trang: 0123456789, dấu ! ? : ; ( ) / -'],
 ['Nghiên cứu văn bản cổ, bia ký và chữ viết.','Điều kiện chụp ảnh khác nhau gây lỗi nhận dạng.','Tư liệu lưu trữ: Phú Yên, Ninh Thuận, Bình Thuận.'],
 ['Tiếng Việt có dấu: ă â ê ô ơ ư đ.','Sắc, huyền, hỏi, ngã, nặng: á à ả ã ạ.','Chữ HOA, chữ thường và số 2026.']
]
def vietnamese_canvas(lines,fontfile='arial.ttf'):
    font=ImageFont.truetype('C:/Windows/Fonts/'+fontfile,35)
    n=Image.new('RGB',(1150,90+len(lines)*78),'white');dr=ImageDraw.Draw(n)
    for i,txt in enumerate(lines):dr.text((35,38+i*78),txt,font=font,fill=(16,16,16))
    return n
for i in range(6):
    lines=texts[i%3]
    f=fontfiles[i%3]
    im=vietnamese_canvas(lines,f)
    if i==3:im=im.filter(ImageFilter.GaussianBlur(0.65))
    if i==4:im=ImageEnhance.Contrast(im).enhance(0.68)
    if i==5:im=im.rotate(-3,expand=True,fillcolor='white')
    add('vietnamese','vietnamese_%02d'%i,im,lines,'vi_only',f,'clear' if i<3 else 'degraded')
symbol=json.loads((ROOT/'ocr-benchmark/data/frozen_suite/symbol_frozen_14_lines.json').read_text(encoding='utf-8'))
for i in range(4):
    vi=texts[i%3][i%3]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',34)
    cham=symbol[20+i*8]
    crop=Image.open(ROOT/'ocr-benchmark/data/frozen_suite'/cham['crop_file']).convert('RGB')
    cw=max(1100,crop.width+60); ch=260
    canvas=Image.new('RGB',(cw,ch),'white'); dr=ImageDraw.Draw(canvas)
    if i%2==0:
        dr.text((25,35),vi,font=font,fill=(10,10,10))
        canvas.paste(crop,(25,140));gt=[vi,cham['gt']]
    else:
        canvas.paste(crop,(25,30))
        dr.text((25,155),vi,font=font,fill=(10,10,10))
        gt=[cham['gt'],vi]
    add('vietnamese','mixed_vi_cham_%02d'%i,canvas,gt,'mixed_vi_cham',
        cham['sample_id'],'vi+Cham synthetic compositing')
obj={'created':'2026-11-22','method':'deterministic image transformations and mixed-script synthetic composites',
     'notes':'GT held constant under non-text-altering transformations; real-camera originals are not independent new photographs.',
     'total':len(cases),'counts':{k:sum(x['suite']==k for x in cases) for k in ['stress','vietnamese']},'cases':cases}
(OUT/'manifest.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
print('GENERATED',obj['counts'],'files',len(cases),flush=True)
