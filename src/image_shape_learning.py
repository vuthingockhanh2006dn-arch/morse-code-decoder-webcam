"""Supervised hand shape learning: labeled photos -> MediaPipe -> SVM/RF.

Browser landmarks are only the feature extractor at inference. No geometric
threshold assigns a gesture label; the classifier learns all four labels.
"""
import hashlib
import json
from pathlib import Path
import time

import cv2
import joblib
import numpy as np
from config import ROOT_DIR, POSE_LABELS

MODEL = ROOT_DIR / "models/image_shape_model.joblib"
REPORT = ROOT_DIR / "reports/image_learning/shape_metrics.json"
VERSION = "pairwise-2d-palm-v1"


def shape_features(landmarks, width=1, height=1):
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Landmark phải gồm 21 điểm hữu hạn x,y,z")
    if not 0 < width <= 8192 or not 0 < height <= 8192:
        raise ValueError("Kích thước ảnh không hợp lệ")
    xy = points[:, :2] * [1., height / width]
    scale = max(np.linalg.norm(xy[0]-xy[9]), np.linalg.norm(xy[5]-xy[17]))
    if scale < 1e-6:
        raise ValueError("Không xác định được kích thước lòng bàn tay")
    # Pairwise distances retain finger shape, remove translation, scale,
    # in-plane rotation and mirroring. Ignore noisy estimated depth.
    xy = (xy-xy[0])/scale
    return np.asarray([np.linalg.norm(xy[a]-xy[b]) for a in range(21)
                       for b in range(a+1, 21)])


def train_shape_model(image_report=None):
    import mediapipe as mp
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

    if image_report is None:
        image_report = json.loads((REPORT.parent / "metrics.json").read_text(encoding="utf-8"))
    assignment = image_report["group_assignment"]
    split_by = image_report["split_by"]
    root = ROOT_DIR / "data/images/morse"
    cache = ROOT_DIR / "data/processed/image_shape"
    cache.mkdir(parents=True, exist_ok=True)
    rows, failures, seen = [], [], set()
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
             min_detection_confidence=.4, model_complexity=1) as hands:
        for path in sorted(root.glob("*/*/*/*.png")):
            subject, session, label, _ = path.relative_to(root).parts
            if label not in POSE_LABELS:
                continue
            group = subject if split_by == "subject" else subject+"/"+session
            if group not in assignment:
                continue
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("Ảnh lỗi: "+str(path))
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            entry = cache/(digest+".json")
            if entry.exists():
                points = json.loads(entry.read_text())
            else:
                h, w = image.shape[:2]
                image = cv2.resize(image, (round(w*400/max(w,h)), round(h*400/max(w,h))))
                result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                points = [[p.x,p.y,p.z] for p in result.multi_hand_landmarks[0].landmark] if result.multi_hand_landmarks else None
                entry.write_text(json.dumps(points))
            if points is None:
                failures.append(str(path.relative_to(root)))
                continue
            h, w = image.shape[:2]
            rows.append((shape_features(points,w,h),label,assignment[group],str(path.relative_to(root))))
    parts = {s:[r for r in rows if r[2] == s] for s in ("train","val","test")}
    for s, part in parts.items():
        if {r[1] for r in part} != set(POSE_LABELS):
            raise ValueError("Tập hình dạng thiếu nhãn: "+s)
    X = {s:np.stack([r[0] for r in part]) for s,part in parts.items()}
    y = {s:np.asarray([r[1] for r in part]) for s,part in parts.items()}
    candidates = [("Shape SVM C="+str(c),make_pipeline(StandardScaler(),SVC(C=c,probability=True,
                   class_weight="balanced", random_state=42))) for c in (1.,10.,100.)]
    candidates.append(("Shape Random Forest",RandomForestClassifier(n_estimators=200,
                       class_weight="balanced",random_state=42,n_jobs=-1)))
    validation=[]; best=None; best_score=-1
    # Webcam calibration remains train-only. Give these scarce examples more
    # weight than the much larger public set; never weight validation/test.
    weights=np.asarray([1. if r[3].startswith('public_digits/') else 4. for r in parts['train']])
    for name,model in candidates:
        kwargs={'svc__sample_weight':weights} if name.startswith('Shape SVM') else {'sample_weight':weights}
        model.fit(X['train'], y['train'], **kwargs)
        pred=model.predict(X['val']); score=float(f1_score(y['val'],pred,average='macro'))
        validation.append({'model':name,'accuracy':float(accuracy_score(y['val'],pred)),'macro_f1':score})
        if score>best_score:
            best=(name,model);best_score=score
    name,model=best
    prob=model.predict_proba(X['val']); confidence=prob.max(axis=1)
    predicted=model.classes_[prob.argmax(axis=1)]
    threshold=.65
    for t in (.55,.6,.65,.7,.75,.8,.85):
        accept=confidence>=t
        if accept.mean()>=.5 and (predicted[accept]==y['val'][accept]).mean()>=.97:
            threshold=t;break
    pred=model.predict(X['test']); accept=model.predict_proba(X['test']).max(axis=1)>=threshold
    report={'model':name,'feature_version':VERSION,'input':'labeled images -> MediaPipe landmarks -> 210 normalized pairwise distances -> supervised classifier',
        'split_counts':{s:len(p) for s,p in parts.items()},'validation':validation,'webcam_train_sample_weight':4.,
        'threshold':threshold,'test_accuracy':float(accuracy_score(y['test'],pred)),
        'test_macro_f1':float(f1_score(y['test'],pred,average='macro')),'test_coverage':float(accept.mean()),
        'confusion_matrix':confusion_matrix(y['test'],pred,labels=POSE_LABELS).tolist(),'labels':POSE_LABELS,
        'classification_report':classification_report(y['test'],pred,output_dict=True,zero_division=0),
        'undetected_images':failures,'group_assignment':assignment,'calibration_train_only_groups':image_report.get('calibration_train_only_groups',[]),
        'limitations':'Conditional on MediaPipe finding a hand. Filename groups are not verified subjects. Webcam performance requires a separate session. Pairwise 2D features do not guarantee invariance to out-of-plane rotation.'}
    artifact={'version':VERSION,'model':model,'threshold':threshold,'report':report}
    MODEL.parent.mkdir(exist_ok=True)
    temporary=MODEL.with_suffix('.tmp');joblib.dump(artifact,temporary);temporary.replace(MODEL)
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (REPORT.parent/'shape_split_manifest.json').write_text(json.dumps([{'path':r[3],'label':r[1],'split':r[2]} for r in rows],indent=2),encoding='utf-8')
    return report


class ImageShapeClassifier:
    def __init__(self):
        self.artifact=None
        self.error=None
        self.reload()

    def reload(self):
        self.artifact=None
        self.error=None
        if MODEL.exists():
            try:
                value=joblib.load(MODEL)
                if value['version'] != VERSION or set(value['model'].classes_) != set(POSE_LABELS):
                    raise ValueError('Mô hình hình dạng không khớp nhãn/phiên bản')
                self.artifact=value
            except Exception as exc:
                self.error=str(exc)

    def predict(self, landmarks, width, height):
        if self.artifact is None:
            raise ValueError('Chưa có mô hình hình dạng học từ ảnh')
        start=time.perf_counter();artifact=self.artifact
        model=artifact['model']; probs=model.predict_proba([shape_features(landmarks,width,height)])[0]
        index=int(probs.argmax());label=str(model.classes_[index]);confidence=float(probs[index])
        accepted=confidence>=artifact['threshold']
        return {'pose':('IDLE' if label=='UNKNOWN' else label) if accepted else 'UNKNOWN',
            'predicted_class':label,'accepted':bool(accepted),'confidence':confidence,
            'confidence_threshold':artifact['threshold'],'recognition_model':artifact['report']['model'],
            'probabilities':{str(c):float(p) for c,p in zip(model.classes_,probs)},'latency_ms':(time.perf_counter()-start)*1000}


if __name__=='__main__':
    report=train_shape_model()
    print(json.dumps({k:v for k,v in report.items() if k not in ('undetected_images','group_assignment')},ensure_ascii=False,indent=2))
