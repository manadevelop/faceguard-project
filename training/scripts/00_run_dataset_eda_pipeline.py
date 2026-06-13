#!/usr/bin/env python3
"""
FaceGuard - Pipeline integral de preparación de datos, EDA y generación de crops.

Este script indexa datasets públicos, infiere etiquetas LIVE/SPOOF, valida
metadata, genera reportes EDA, crea splits estratificados y prepara las
carpetas finales para entrenamiento PyTorch.
"""
from __future__ import annotations
import argparse, hashlib, json, math, shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT=Path(__file__).resolve().parents[2]
TRAINING_DIR=PROJECT_ROOT/'training'; DATA_DIR=TRAINING_DIR/'data'
RAW_DIR=DATA_DIR/'raw'; INTERIM_DIR=DATA_DIR/'interim'; METADATA_DIR=DATA_DIR/'metadata'
CROPS_DIR=DATA_DIR/'crops'; FRAMES_DIR=DATA_DIR/'frames'
PROCESSED_DIR=DATA_DIR/'processed'; PROCESSED_DEPTH_DIR=DATA_DIR/'processed_depth'
OUTPUTS_DIR=TRAINING_DIR/'outputs'; EDA_DIR=OUTPUTS_DIR/'eda'; REPORTS_DIR=OUTPUTS_DIR/'reports'
IMG_EXT={'.jpg','.jpeg','.png','.bmp','.webp'}; VID_EXT={'.mp4','.avi','.mov','.mkv','.webm','.mpeg','.mpg'}

@dataclass
# Configuración del pipeline de datos: tamaños, splits, calidad y rutas.
class PipelineConfig:
    raw_dir:Path=RAW_DIR
    image_size:int=224
    seed:int=42
    test_size:float=.15
    val_size:float=.15
    max_frames_per_video:int=16
    blur_threshold:float=25.
    min_brightness:float=25.
    max_brightness:float=235.
    jpeg_quality:int=92
    fallback_center_crop:bool=True

# Crea todas las carpetas usadas por metadata, crops, reportes y outputs.
def ensure_dirs():
    for d in [RAW_DIR,INTERIM_DIR,METADATA_DIR,CROPS_DIR,FRAMES_DIR,PROCESSED_DIR,PROCESSED_DEPTH_DIR,OUTPUTS_DIR,EDA_DIR,REPORTS_DIR]:
        d.mkdir(parents=True,exist_ok=True)

# Devuelve una ruta relativa al proyecto para guardar metadata portable.
def rel(p:Path)->str:
    try:return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:return str(p)

# Hash corto para crear identificadores reproducibles a partir de rutas.
def md5_short(s:str)->str:return hashlib.md5(s.encode()).hexdigest()[:12]
# Normaliza texto para comparar nombres de carpetas/archivos sin depender de mayúsculas.
def nt(s:str)->str:return s.lower().strip().replace(' ','_').replace('-','_')
# Genera tokens desde ruta y nombre de archivo para inferir dataset, modalidad y etiqueta.
def toks(p:Path)->List[str]:
    out=[]
    for part in p.parts:
        x=nt(part); out.append(x); out += [q for q in x.split('_') if q]
    x=nt(p.stem); out.append(x); out += [q for q in x.split('_') if q]
    return out

# Infere a qué dataset pertenece una muestra según los tokens de su ruta.
def infer_dataset(p:Path)->str:
    t=set(toks(p))
    if {'casia_fasd','casiafasd','casia'} & t:return 'CASIA-FASD'
    if {'anti_spoofing','antispoofing'} & t:return 'Anti-Spoofing'
    if {'celeba_spoof','celebaspoof'} & t:return 'CelebA-Spoof'
    if {'oulu_npu','oulu'} & t:return 'OULU-NPU'
    if 'replay_attack' in t:return 'Replay-Attack'
    return 'Unknown'

# Infere modalidad RGB/depth; en CASIA se detecta depth por nombre de ruta.
def infer_modality(p:Path)->str:
    t=set(toks(p))
    if {'casia_fasd','casiafasd','casia'} & t and 'depth' in t:return 'depth'
    return 'rgb'

# Infere etiqueta binaria LIVE=1/SPOOF=0 y tipo de ataque desde la ruta.
def infer_label(p:Path)->Tuple[Optional[int],str,str,bool]:
    t=set(toks(p)); stem=nt(p.stem)
    if {'casia_fasd','casiafasd','casia'} & t:
        if stem.endswith('_real') or '_real' in stem or 'real' in t:return 1,'none','real',True
        if stem.endswith('_fake') or '_fake' in stem or 'fake' in t:return 0,'unknown','fake',True
    if {'anti_spoofing','antispoofing'} & t:
        if 'live_selfie' in t or 'live_video' in t or 'live' in t:return 1,'none','live',True
        if 'replay' in t:return 0,'replay','replay',True
        if 'printouts' in t or 'printout' in t:return 0,'print','printouts',True
        if 'cut' in t and 'out' in t:return 0,'cut_photo','cut-out printouts',True
    if {'real','live','genuine'} & t:return 1,'none','live',True
    if {'fake','spoof','attack'} & t:return 0,'unknown','spoof',True
    if {'print','printed','printout','printouts'} & t:return 0,'print','print',True
    if 'screen' in t:return 0,'screen','screen',True
    if 'replay' in t:return 0,'replay','replay',True
    return None,'unknown','unmapped',False

# Construye un identificador de sujeto aproximado para análisis y splits.
def subject_id(p:Path)->str:
    ds=infer_dataset(p); stem=nt(p.stem)
    if ds=='CASIA-FASD':return f"casia_subject_{stem.split('_')[0]}"
    if ds=='Anti-Spoofing':return f"anti_{md5_short(rel(p.parent))}"
    return f"subject_{md5_short(rel(p.parent))}"

# Lee metadatos básicos de imagen: tamaño, brillo y blur para control de calidad.
def img_info(p:Path):
    try:
        with Image.open(p) as im:
            im=ImageOps.exif_transpose(im).convert('RGB'); w,h=im.size
            a=np.asarray(im); g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
            return w,h,True,'',float(g.mean()),float(cv2.Laplacian(g,cv2.CV_64F).var())
    except Exception as e:return None,None,False,str(e)[:200],None,None

# Lee metadatos básicos de video: tamaño, frames y FPS.
def vid_info(p:Path):
    cap=cv2.VideoCapture(str(p))
    if not cap.isOpened():return None,None,None,None,False,'cannot_open_video'
    try:return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None,float(cap.get(cv2.CAP_PROP_FPS)) or None,True,''
    finally:cap.release()

# Paso 1: indexa archivos raw y construye metadata_raw.csv con etiquetas inferidas.
def build_metadata(cfg):
    print('\n[1/7] Construyendo metadata_raw.csv ...'); ensure_dirs()
    files=sorted([p for p in cfg.raw_dir.rglob('*') if p.is_file() and p.suffix.lower() in (IMG_EXT|VID_EXT)])
    rows=[]
    for i,p in enumerate(tqdm(files,desc='Indexando archivos'),1):
        st='image' if p.suffix.lower() in IMG_EXT else 'video'
        ds=infer_dataset(p); mod=infer_modality(p); bl,atk,orig,lv=infer_label(p)
        w=h=fc=fps=br=blur=None; ok=True; err=''
        if st=='image':w,h,ok,err,br,blur=img_info(p)
        else:w,h,fc,fps,ok,err=vid_info(p)
        rows.append(dict(sample_id=f'SMP_{i:08d}_{md5_short(rel(p))}',dataset_name=ds,source_type=st,modality=mod,file_path=rel(p),crop_path='',processed_path='',processed_depth_path='',frame_path='',video_id=p.stem if st=='video' else '',frame_index='',pair_key=p.name if ds=='CASIA-FASD' else '',subject_id=subject_id(p),original_label=orig,binary_label='' if bl is None else bl,attack_type=atk,split='',width=w or '',height=h or '',frame_count=fc or '',fps=fps or '',file_size=p.stat().st_size,brightness='' if br is None else round(br,4),blur_score='' if blur is None else round(blur,4),face_detected='',num_faces='',face_bbox_x='',face_bbox_y='',face_bbox_w='',face_bbox_h='',face_quality_score='',is_readable=ok,label_valid=lv,is_low_quality=False,is_valid=bool(ok and lv and bl in (0,1)),error=err))
    df=pd.DataFrame(rows); out=METADATA_DIR/'metadata_raw.csv'; df.to_csv(out,index=False); print(f'OK: {rel(out)} | muestras={len(df)}'); return df

# Paso 2: registra errores de etiquetas o archivos no legibles.
def validate_labels(df):
    print('\n[2/7] Validando labels ...'); ensure_dirs(); errs=[]
    for _,r in df.iterrows():
        reason=[]
        if str(r.get('is_readable','')).lower()!='true':reason.append('archivo_no_legible')
        if str(r.get('label_valid','')).lower()!='true':reason.append('label_no_mapeado')
        if str(r.get('binary_label','')) not in {'0','1','0.0','1.0'}:reason.append('binary_label_invalido')
        if reason:
            x=r.to_dict(); x['validation_errors']='|'.join(reason); errs.append(x)
    pd.DataFrame(errs).to_csv(INTERIM_DIR/'label_errors.csv',index=False)
    pd.DataFrame([{'source':'CASIA-FASD','rule':'*_real.jpg','binary_label':1},{'source':'CASIA-FASD','rule':'*_fake.jpg','binary_label':0},{'source':'anti_spoofing','rule':'live_selfie/live_video','binary_label':1},{'source':'anti_spoofing','rule':'printouts/cut-out printouts/replay','binary_label':0}]).to_csv(INTERIM_DIR/'label_mapping.csv',index=False)
    (METADATA_DIR/'data_dictionary.md').write_text('LIVE=1, SPOOF=0. Depth se recorta usando bounding box del RGB emparejado.\n',encoding='utf-8')
    print(f'OK: errores={len(errs)} | {rel(INTERIM_DIR/"label_errors.csv")}')

# Guarda gráfico de barras y su CSV asociado para el EDA.
def save_bar(s,title,out):
    if len(s)==0:return
    plt.figure(figsize=(10,4.8)); s.plot(kind='bar'); plt.title(title); plt.tight_layout()
    out.parent.mkdir(parents=True,exist_ok=True); plt.savefig(out,dpi=160); plt.close()
    s.reset_index().to_csv(out.with_suffix('.csv'),index=False)

# Guarda histograma de variables numéricas del dataset.
def save_hist(v,title,out,xlabel):
    a=pd.to_numeric(v,errors='coerce').dropna()
    if len(a)==0:return
    plt.figure(figsize=(10,4.8)); plt.hist(a,bins=min(60,max(10,int(math.sqrt(len(a)))))); plt.title(title); plt.xlabel(xlabel); plt.tight_layout()
    out.parent.mkdir(parents=True,exist_ok=True); plt.savefig(out,dpi=160); plt.close()

# Paso 3: genera distribución de clases, fuentes, modalidades y resumen EDA.
def run_eda(df,stage='raw'):
    print('\n[3/7] Generando EDA completo ...')
    labels=df['binary_label'].replace({1:'LIVE',0:'SPOOF','1':'LIVE','0':'SPOOF','':'UNMAPPED'}).fillna('UNMAPPED')
    save_bar(labels.value_counts(),'Distribución LIVE/SPOOF',EDA_DIR/'class_distribution.png')
    for c,n in [('attack_type','attack_type_distribution.png'),('dataset_name','dataset_distribution.png'),('source_type','source_type_distribution.png'),('modality','modality_distribution.png'),('split','split_distribution.png')]:
        if c in df:save_bar(df[c].fillna('unknown').replace('', 'unsplit').value_counts(),c,EDA_DIR/n)
    if 'subject_id' in df:save_bar(df['subject_id'].fillna('unknown').value_counts().head(30),'Top sujetos',EDA_DIR/'subject_distribution.png')
    for c in ['width','height','file_size','brightness','blur_score','frame_count','fps']:
        if c in df:save_hist(df[c],c,EDA_DIR/f'{c}_distribution.png',c)
    lines=['# EDA Summary - FaceGuard','',f'- Etapa: `{stage}`',f'- Total muestras: **{len(df)}**','']
    for c in ['dataset_name','source_type','modality','binary_label','attack_type','split','is_valid']:
        if c in df:
            lines += [f'## {c}','']; lines += [f'- `{k}`: {v}' for k,v in df[c].fillna('NA').astype(str).value_counts().items()]; lines.append('')
    (EDA_DIR/'eda_summary.md').write_text('\n'.join(lines),encoding='utf-8'); df.describe(include='all').transpose().to_csv(EDA_DIR/'eda_describe.csv')
    print(f'OK: EDA en {rel(EDA_DIR)}')

# Paso 4: filtra muestras inválidas y marca baja calidad por blur/brillo.
def clean_dataset(df,cfg):
    print('\n[4/7] Limpiando dataset ...'); df=df.copy(); df['binary_label']=pd.to_numeric(df['binary_label'],errors='coerce')
    valid=(df['is_readable'].astype(str).str.lower()=='true')&(df['label_valid'].astype(str).str.lower()=='true')&df['binary_label'].isin([0,1])
    br=pd.to_numeric(df['brightness'],errors='coerce'); blur=pd.to_numeric(df['blur_score'],errors='coerce')
    low=(df['source_type'].eq('image')&df['modality'].eq('rgb')&(((blur.notna())&(blur<cfg.blur_threshold))|((br.notna())&((br<cfg.min_brightness)|(br>cfg.max_brightness)))))
    df['is_low_quality']=low; df[low].to_csv(INTERIM_DIR/'low_quality_files.csv',index=False)
    out=df[valid].copy(); out.to_csv(METADATA_DIR/'metadata_clean.csv',index=False)
    pd.DataFrame([{'before':len(df),'after':len(out),'removed':len(df)-len(out),'low_quality_flagged_not_removed':int(low.sum())}]).to_csv(INTERIM_DIR/'cleaning_log.csv',index=False)
    print(f'OK: {rel(METADATA_DIR/"metadata_clean.csv")} | antes={len(df)} después={len(out)}'); return out

# Paso 5: crea splits train/val/test estratificados por etiqueta.
def create_splits(df,cfg):
    print('\n[5/7] Creando splits train/val/test ...'); d=df.copy().reset_index(drop=True); d['binary_label']=pd.to_numeric(d['binary_label']).astype(int)
    tr,tmp=train_test_split(d,test_size=cfg.test_size+cfg.val_size,random_state=cfg.seed,stratify=d['binary_label'])
    val,test=train_test_split(tmp,test_size=cfg.test_size/(cfg.test_size+cfg.val_size),random_state=cfg.seed,stratify=tmp['binary_label'])
    d['split']=''; d.loc[tr.index,'split']='train'; d.loc[val.index,'split']='val'; d.loc[test.index,'split']='test'
    d.to_csv(METADATA_DIR/'metadata_processed.csv',index=False)
    for s in ['train','val','test']: d[d['split']==s].to_csv(METADATA_DIR/f'{s}.csv',index=False)
    print('OK splits:',d['split'].value_counts().to_dict(),'| por label:',d.groupby(['split','binary_label']).size().to_dict()); return d

# Carga el detector Haar Cascade usado para localizar rostros en imágenes RGB.
def detector():return cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
# Lee una imagen aplicando corrección EXIF y conversión RGB.
def read_img(p):return np.asarray(ImageOps.exif_transpose(Image.open(p).convert('RGB')))
# Detecta el rostro principal y devuelve bounding box.
def detect(img,det):
    faces=det.detectMultiScale(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY),1.1,5,minSize=(40,40))
    if faces is None or len(faces)==0:return None,0
    faces=sorted(faces,key=lambda b:int(b[2])*int(b[3]),reverse=True); return tuple(map(int,faces[0])),len(faces)
# Recorta el rostro y lo redimensiona al tamaño esperado por entrenamiento.
def crop(img,bbox,size,fallback=True):
    h,w=img.shape[:2]
    if bbox is None:
        if not fallback:raise ValueError('no_face_detected')
        side=min(w,h); x=(w-side)//2; y=(h-side)//2; c=img[y:y+side,x:x+side]
    else:
        x,y,bw,bh=bbox; m=int(max(bw,bh)*.25); c=img[max(0,y-m):min(h,y+bh+m), max(0,x-m):min(w,x+bw+m)]
    if c.size==0:raise ValueError('empty_crop')
    return cv2.resize(c,(size,size),interpolation=cv2.INTER_AREA)
# Calcula métricas simples de calidad del crop: brillo, blur y área facial.
def quality(img,bbox,cfg):
    g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY); blur=float(cv2.Laplacian(g,cv2.CV_64F).var()); bright=float(g.mean())
    ok=blur>=cfg.blur_threshold and cfg.min_brightness<=bright<=cfg.max_brightness and (bbox is not None or cfg.fallback_center_crop)
    return round(min(1,blur/max(cfg.blur_threshold*4,1))*.35+(1-min(abs(bright-128)/128,1))*.3,4),ok,{'blur_score':round(blur,4),'brightness':round(bright,4)}
# Guarda imagen procesada con calidad JPEG controlada.
def save(img,p,q):p.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(img).save(p,quality=q)
# Procesa una muestra RGB: detección facial, crop y registro de ruta procesada.
def process_rgb(r,det,cfg,bc,qc):
    row=r.to_dict(); src=PROJECT_ROOT/row['file_path']; cls='live' if int(float(row['binary_label']))==1 else 'spoof'; dst=CROPS_DIR/row['split']/cls/f"{row['sample_id']}.jpg"
    try:
        img=read_img(src); bbox,n=detect(img,det); cr=crop(img,bbox,cfg.image_size,cfg.fallback_center_crop); q,ok,qi=quality(img,bbox,cfg); save(cr,dst,cfg.jpeg_quality)
        row.update(crop_path=rel(dst),face_detected=bbox is not None,num_faces=n,face_quality_score=q,is_low_quality=not ok,blur_score=qi['blur_score'],brightness=qi['brightness'],is_valid=True,error='')
        if bbox:
            x,y,w,h=bbox; row.update(face_bbox_x=x,face_bbox_y=y,face_bbox_w=w,face_bbox_h=h)
        if row['dataset_name']=='CASIA-FASD' and row['modality']=='rgb': bc[row['pair_key']]=bbox; qc[row['pair_key']]=row.copy()
    except Exception as e: row.update(is_valid=False,error=str(e)[:200],face_detected=False,num_faces=0)
    return row
# Procesa depth usando el bounding box correspondiente de la imagen RGB.
def process_depth(r,cfg,bc,qc):
    row=r.to_dict(); src=PROJECT_ROOT/row['file_path']; cls='live' if int(float(row['binary_label']))==1 else 'spoof'; dst=CROPS_DIR/row['split']/'depth'/cls/f"{row['sample_id']}.jpg"
    try:
        img=read_img(src); bbox=bc.get(row['pair_key']); cr=crop(img,bbox,cfg.image_size,cfg.fallback_center_crop); save(cr,dst,cfg.jpeg_quality)
        qr=qc.get(row['pair_key'],{}); row.update(crop_path=rel(dst),face_detected=bool(qr.get('face_detected',bbox is not None)),num_faces=int(qr.get('num_faces',1 if bbox else 0)),face_quality_score=qr.get('face_quality_score',1 if bbox else .5),is_low_quality=bool(qr.get('is_low_quality',False)),is_valid=True,error='' if bbox else 'depth_crop_used_center_fallback_no_rgb_bbox')
        if bbox:
            x,y,w,h=bbox; row.update(face_bbox_x=x,face_bbox_y=y,face_bbox_w=w,face_bbox_h=h)
    except Exception as e: row.update(is_valid=False,error=str(e)[:200],face_detected=False,num_faces=0)
    return row
# Extrae frames de videos y genera crops faciales para entrenamiento.
def extract_frames(r,det,cfg):
    rows=[]; src=PROJECT_ROOT/r['file_path']; cap=cv2.VideoCapture(str(src))
    if not cap.isOpened(): x=r.to_dict(); x.update(is_valid=False,error='cannot_open_video'); return [x]
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0; idxs=list(range(cfg.max_frames_per_video)) if total<=0 else np.linspace(0,max(total-1,0),min(cfg.max_frames_per_video,total),dtype=int).tolist()
    cls='live' if int(float(r['binary_label']))==1 else 'spoof'
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(fi)); ok,bgr=cap.read(); row=r.to_dict(); row['sample_id']=f"{r['sample_id']}_F{int(fi):06d}"; row['frame_index']=int(fi); row['source_type']='frame'; row['modality']='rgb'
        if not ok or bgr is None:row.update(is_valid=False,error='frame_read_error'); rows.append(row); continue
        img=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); fp=FRAMES_DIR/r['split']/cls/r['video_id']/f"frame_{int(fi):06d}.jpg"; cp=CROPS_DIR/r['split']/cls/r['video_id']/f"{row['sample_id']}.jpg"
        try:
            bbox,n=detect(img,det); cr=crop(img,bbox,cfg.image_size,cfg.fallback_center_crop); q,okq,qi=quality(img,bbox,cfg)
            save(img,fp,cfg.jpeg_quality); save(cr,cp,cfg.jpeg_quality)
            row.update(frame_path=rel(fp),crop_path=rel(cp),face_detected=bbox is not None,num_faces=n,face_quality_score=q,is_low_quality=not okq,blur_score=qi['blur_score'],brightness=qi['brightness'],width=img.shape[1],height=img.shape[0],is_valid=True,error='')
        except Exception as e:row.update(is_valid=False,error=str(e)[:200])
        rows.append(row)
    cap.release(); return rows
# Paso 6: aplica recorte facial a imágenes y videos válidos.
def detect_crop_faces(df,cfg):
    print('\n[6/7] Detectando/recortando rostros y extrayendo frames de videos ...'); det=detector(); bc={}; qc={}; rows=[]
    rgb=df[(df.source_type=='image')&(df.modality=='rgb')]; vids=df[df.source_type=='video']; dep=df[(df.source_type=='image')&(df.modality=='depth')]
    for _,r in tqdm(rgb.iterrows(),total=len(rgb),desc='Procesando RGB'): rows.append(process_rgb(r,det,cfg,bc,qc))
    for _,r in tqdm(vids.iterrows(),total=len(vids),desc='Procesando videos'): rows+=extract_frames(r,det,cfg)
    for _,r in tqdm(dep.iterrows(),total=len(dep),desc='Procesando depth con bbox RGB'): rows.append(process_depth(r,cfg,bc,qc))
    out=pd.DataFrame(rows); out.to_csv(METADATA_DIR/'metadata_processed.csv',index=False)
    miss=out[out.crop_path.fillna('').astype(str)=='']; miss.to_csv(INTERIM_DIR/'missing_faces.csv',index=False)
    low=out[out.is_low_quality.astype(str).str.lower()=='true']; low.to_csv(INTERIM_DIR/'low_quality_faces.csv',index=False)
    print(f'OK crops/frames | filas procesadas={len(out)} | sin crop={len(miss)} | low_quality={len(low)}'); return out
# Paso 7: organiza crops en carpetas PyTorch train/val/test/live/spoof.
def prepare_processed_dataset(df,cfg):
    print('\n[7/7] Preparando training/data/processed y processed_depth ...')
    for base in [PROCESSED_DIR,PROCESSED_DEPTH_DIR]:
        if base.exists():shutil.rmtree(base)
        for s in ['train','val','test']:
            for c in ['live','spoof']:(base/s/c).mkdir(parents=True,exist_ok=True)
    rows=[]
    for _,r in tqdm(df.iterrows(),total=len(df),desc='Copiando crops'):
        row=r.to_dict(); crop_path=str(row.get('crop_path','')); valid=str(row.get('is_valid','')).lower() in {'true','1'}
        if not valid or not crop_path: rows.append(row); continue
        src=PROJECT_ROOT/crop_path; cls='live' if int(float(row['binary_label']))==1 else 'spoof'; base=PROCESSED_DEPTH_DIR if row.get('modality')=='depth' else PROCESSED_DIR; dst=base/row['split']/cls/src.name
        if src.exists(): shutil.copy2(src,dst); row['processed_depth_path' if row.get('modality')=='depth' else 'processed_path']=rel(dst)
        rows.append(row)
    final=pd.DataFrame(rows); final.to_csv(METADATA_DIR/'metadata_processed.csv',index=False)
    for s in ['train','val','test']: final[final.split==s].to_csv(METADATA_DIR/f'{s}.csv',index=False)
    pairs=[]; casia=final[final.dataset_name.astype(str)=='CASIA-FASD']
    for key,g in casia.groupby('pair_key'):
        rgb=g[g.modality.astype(str)=='rgb']; dep=g[g.modality.astype(str)=='depth']
        pairs.append({'pair_key':key,'has_rgb':len(rgb)>0,'has_depth':len(dep)>0,'rgb_path':str(rgb.iloc[0].get('processed_path','')) if len(rgb) else '','depth_path':str(dep.iloc[0].get('processed_depth_path','')) if len(dep) else '','binary_label':str(g.iloc[0].get('binary_label','')),'split':str(g.iloc[0].get('split',''))})
    pd.DataFrame(pairs).to_csv(METADATA_DIR/'casia_rgb_depth_pairs.csv',index=False)
    counts={name:{s:{c:len(list((base/s/c).glob('*.jpg'))) for c in ['live','spoof']} for s in ['train','val','test']} for name,base in [('rgb',PROCESSED_DIR),('depth',PROCESSED_DEPTH_DIR)]}
    (REPORTS_DIR/'processed_counts.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    print('OK processed:',counts); print(f'OK pares RGB-depth CASIA: {len(pairs)} | {rel(METADATA_DIR/"casia_rgb_depth_pairs.csv")}'); return final
# Genera reporte final de preparación de datos y conteos principales.
def write_report(cfg,final):
    counts=final.groupby(['split','modality','binary_label']).size().reset_index(name='n') if len(final) else pd.DataFrame()
    txt=['# FaceGuard - Dataset + EDA Pipeline Report','','## Configuración','```json',json.dumps({k:str(v) for k,v in asdict(cfg).items()},indent=2,ensure_ascii=False),'```','','## Conteo final',counts.to_string(index=False) if len(counts) else 'Sin datos.']
    (REPORTS_DIR/'dataset_eda_pipeline_report.md').write_text('\n'.join(txt),encoding='utf-8')
# Punto de entrada: ejecuta todo el pipeline de datos en orden reproducible.
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-dir',default=str(RAW_DIR)); ap.add_argument('--image-size',type=int,default=224); ap.add_argument('--max-frames-per-video',type=int,default=16); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--no-fallback-center-crop',action='store_true')
    args=ap.parse_args(); cfg=PipelineConfig(raw_dir=Path(args.raw_dir),image_size=args.image_size,max_frames_per_video=args.max_frames_per_video,seed=args.seed,fallback_center_crop=not args.no_fallback_center_crop)
    ensure_dirs(); raw=build_metadata(cfg); validate_labels(raw); run_eda(raw,'raw'); clean=clean_dataset(raw,cfg); sp=create_splits(clean,cfg); cr=detect_crop_faces(sp,cfg); final=prepare_processed_dataset(cr,cfg); run_eda(final,'processed'); write_report(cfg,final)
    print('\nLISTO. Pipeline completo ejecutado correctamente.'); print(f'Reporte EDA: {rel(EDA_DIR/"eda_summary.md")}'); print(f'Reporte final: {rel(REPORTS_DIR/"dataset_eda_pipeline_report.md")}')
if __name__=='__main__':main()
