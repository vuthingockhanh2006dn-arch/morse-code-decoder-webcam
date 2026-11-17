#!/usr/bin/env python3
"""Capture genuine webcam gesture videos with labels and session metadata."""
import argparse,csv,datetime,pathlib,time,uuid
try:import cv2
except ImportError:raise SystemExit('Install OpenCV: pip install opencv-python')
ROOT=pathlib.Path(__file__).resolve().parents[1]
LABELS=('DOT','DASH','SPACE','CONFIRM','DELETE','UNKNOWN')
def main():
 p=argparse.ArgumentParser();p.add_argument('--person',required=True);p.add_argument('--session',default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));p.add_argument('--seconds',type=float,default=3);p.add_argument('--camera',type=int,default=0);a=p.parse_args()
 cap=cv2.VideoCapture(a.camera)
 if not cap.isOpened():raise SystemExit('Camera unavailable')
 out=ROOT/'data/raw/videos'/a.person/a.session;out.mkdir(parents=True,exist_ok=True)
 meta=ROOT/'metadata/video_manifest.csv';fields=['sample_id','person_id','session_id','label','file','timestamp_utc','frames','fps','seconds']
 print('Keys: 1 DOT, 2 DASH, 3 SPACE, 4 CONFIRM, 5 DELETE, 6 UNKNOWN, q quit')
 try:
  while True:
   ok,frame=cap.read()
   if not ok:break
   cv2.putText(frame,'1 DOT 2 DASH 3 SPACE 4 CONFIRM 5 DELETE 6 UNKNOWN | q quit',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2)
   cv2.imshow('Morse webcam recorder',frame);key=cv2.waitKey(1)&255
   if key==ord('q'):break
   if key not in map(ord,'123456'):continue
   label=LABELS[key-ord('1')];sid=uuid.uuid4().hex[:12];path=out/f'{label}_{sid}.mp4';fps=cap.get(cv2.CAP_PROP_FPS) or 30
   if fps<5 or fps>120:fps=30
   h,w=frame.shape[:2];writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));start=time.monotonic();frames=0
   while time.monotonic()-start<a.seconds:
    ok,f=cap.read()
    if not ok:break
    writer.write(f);frames+=1;cv2.putText(f,f'RECORDING {label}',(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2);cv2.imshow('Morse webcam recorder',f);cv2.waitKey(1)
   writer.release()
   with meta.open('a',newline='',encoding='utf-8') as f:
    wr=csv.DictWriter(f,fieldnames=fields)
    if f.tell()==0:wr.writeheader()
    wr.writerow(dict(sample_id=sid,person_id=a.person,session_id=a.session,label=label,file=path.relative_to(ROOT),timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),frames=frames,fps=fps,seconds=round(frames/fps,2)))
   print('Saved',path)
 finally:cap.release();cv2.destroyAllWindows()
if __name__=='__main__':main()
