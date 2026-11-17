#!/usr/bin/env python3
"""Download a real public image dataset, with integrity checks and metadata.
No images are synthesized or relabeled as Morse gestures.
"""
import argparse,csv,hashlib,json,os,pathlib,random,sys,time,urllib.request,urllib.error
ROOT=pathlib.Path(__file__).resolve().parents[1]
API='https://api.github.com/repos/ardamavi/Sign-Language-Digits-Dataset/git/trees/master?recursive=1'
RAW='https://raw.githubusercontent.com/ardamavi/Sign-Language-Digits-Dataset/master/'
UA={'User-Agent':'Morse-Webcam-Dataset-Collector/1.0','Accept':'application/vnd.github+json'}
def fetch(url,tries=4):
 for n in range(tries):
  try:
   req=urllib.request.Request(url,headers=UA)
   with urllib.request.urlopen(req,timeout=40) as r:return r.read()
  except (urllib.error.URLError,TimeoutError) as e:
   if n==tries-1:raise
   time.sleep(2**n)
def main():
 p=argparse.ArgumentParser();p.add_argument('--min-images',type=int,default=2000);p.add_argument('--max-images',type=int,default=2200);p.add_argument('--seed',type=int,default=42);args=p.parse_args()
 if args.min_images>args.max_images:p.error('min-images must be <= max-images')
 tree=json.loads(fetch(API))['tree']; imgs=[x['path'] for x in tree if x.get('type')=='blob' and x['path'].startswith('Dataset/') and x['path'].lower().endswith(('.jpg','.jpeg','.png'))]
 random.Random(args.seed).shuffle(imgs)
 if len(imgs)<args.min_images:raise SystemExit(f'Source has only {len(imgs)} images; requirement not met')
 imgs=imgs[:args.max_images]; out=ROOT/'data/raw/images/sign_language_digits';out.mkdir(parents=True,exist_ok=True)
 manifest=ROOT/'metadata/image_manifest.csv'; records={}
 if manifest.exists():
  with manifest.open(newline='',encoding='utf-8') as f:
   for row in csv.DictReader(f):records[row['source_path']]=row
 for i,source in enumerate(imgs,1):
  dest=out/source.split('/')[1]/pathlib.Path(source).name;dest.parent.mkdir(parents=True,exist_ok=True)
  try:
   if dest.exists() and dest.stat().st_size>100: data=dest.read_bytes()
   else:
    data=fetch(RAW+source)
    if len(data)<100:raise ValueError('Image too small')
    dest.write_bytes(data)
   if data[:2]!=b'\xff\xd8' and data[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Not a JPEG/PNG file')
   records[source]={'sample_id':hashlib.sha1(source.encode()).hexdigest()[:16],'source_dataset':'Sign Language Digits','source_path':source,'local_path':dest.relative_to(ROOT).as_posix(),'source_url':RAW+source,'original_label':source.split('/')[1],'morse_label':'UNLABELED','media_type':'image','sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'status':'verified'}
  except Exception as e:print(f'FAIL {source}: {e}',file=sys.stderr)
  if i%100==0:print(f'Processed {i}/{len(imgs)}',flush=True)
 fields=['sample_id','source_dataset','source_path','local_path','source_url','original_label','morse_label','media_type','sha256','bytes','status']
 with manifest.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records[k] for k in sorted(records))
 count=sum(r['status']=='verified' for r in records.values());report={'verified_real_images':count,'required_minimum':args.min_images,'requirement_met':count>=args.min_images,'morse_labeled_samples':0,'video_count':len(list((ROOT/'data/raw/videos').rglob('*.mp4'))),'source':'https://github.com/ardamavi/Sign-Language-Digits-Dataset','note':'Source labels are digits 0-9, NOT Morse gesture classes. Videos must be collected separately.'}
 (ROOT/'reports/download_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
 if count<args.min_images:sys.exit(2)
if __name__=='__main__':main()
