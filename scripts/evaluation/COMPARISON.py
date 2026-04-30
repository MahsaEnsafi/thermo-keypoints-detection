import os 
import csv
import cv2
import numpy as np
def create_mask(thrm):
    # Gaussian filter
    gus=cv2.GaussianBlur(thrm,(5,5),0)

    # Otsu threshold
    _,mask=cv2.threshold(gus,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # morphology cleanup
    krnl=np.ones((9,9),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,krnl,iterations=2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,krnl,iterations=1)

    # keep all large components (avoid tiny noise)
    num,labels, stats, _ = cv2.connectedComponentsWithStats(mask,connectivity=8)
    final_mask=np.zeros_like(mask)

    for i in range(1,num):
        area=stats[i,cv2.CC_STAT_AREA]
        if area > 8000:
            final_mask[labels==i]=255

    if final_mask.sum()==0:
        final_mask=mask.copy()

    # ensure body is white
    ins=thrm[final_mask>0]
    outs=thrm[final_mask==0]
    if ins.size and outs.size and ins.mean() < outs.mean():
        final_mask=255-final_mask

    return final_mask
#------------------------------------------------------------------------------------------------------
def enhance_thermogram(thrm):
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    thrm_enhc=clahe.apply(thrm)
    return thrm_enhc
#------------------------------------------------------------------------------------------------

def detection_quality(detect_fn,thrm,mask,model=None):
    if model is None:
        kpts,desc=detect_fn(thrm)
    else:
        kpts,desc=detect_fn(thrm,model)
    if not kpts:
        return 0,0,0.0,[]
    pts=np.array([kp.pt for kp in kpts]).astype(int)
    xs,ys=pts[:,0],pts[:,1]
    h,w=mask.shape
    valid=(xs>=0) & (xs<w) & (ys>=0) & (ys<h)
    inside_mask=mask[ys[valid],xs[valid]]>0
    inside=np.sum(inside_mask)
    total=len(kpts)
    ratio=inside / total if total>0 else 0
    return total,inside,ratio,kpts
#----------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------

def save_results(thrm,total,kpts,inside,patient,view,csv_path,method,ratio,w,h,mask):
    try:
        csv_dir=os.path.dirname(csv_path)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        file_exists=os.path.isfile(csv_path) 
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Method", "Patient", "View", "Total Keypoints",
                    "Inside Keypoints", "Detection Quality"])
            writer.writerow([
            method,
            patient,
            view,
            total,
            inside,
            round(ratio, 4)
        ])
    except Exception as e:
        print("Error waiting to CSV")
    
    try:
        out_path='../experiments/results/'
        fldr_path=os.path.join(out_path,patient)
        if not os.path.exists(fldr_path):
            os.mkdir(fldr_path)
        view_name=method+'_keypoints_'+view
        view_path=os.path.join(fldr_path,view_name)
        img_all_kpt=cv2.drawKeypoints(thrm,kpts,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(view_path,img_all_kpt)
        kpts_in=[]
        for kp in kpts:
            x,y=int(round(kp.pt[0])),int(round(kp.pt[1]))
            if 0<=x<w and 0<=y<h and mask[y,x]>0:
                kpts_in.append(kp)
        img_all_kpt_in=cv2.drawKeypoints(thrm,kpts_in,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        view_name_in=method+'_keypoints_inside_'+view
        view_path_in=os.path.join(fldr_path,view_name_in)
        cv2.imwrite(view_path_in,img_all_kpt_in)
    except Exception as e:
        print('Error saving keypoints therm0ograms')
#---------------------------------------------------------------------------------------------
def  execution(fn,model,name):
    for patient in patients:
        patient_path=os.path.join(path,patient)
        thermograms=os.listdir(patient_path)
        for view in thermograms:
            view_path=os.path.join(patient_path,view)
            thrm=cv2.imread(view_path)
            if len(thrm.shape)==3:
                thrm=cv2.cvtColor(thrm,cv2.COLOR_BGR2GRAY)
            width=thrm.shape[1]
            height=thrm.shape[0]
            mask=create_mask(thrm)
            thrm_enh=enhance_thermogram(thrm)
            total,inside,ratio,kpts=detection_quality(fn,thrm_enh,mask,model)
            save_results(thrm,total,kpts,inside,patient,view,csv_path,name,ratio,width,height,mask)

out_path='../experiments/results/'
csv_path = "results.csv"
csv_path=os.path.join(out_path,csv_path)
file_exists = os.path.isfile(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Method", "Patient", "View", "Total Keypoints",
                    "Inside Keypoints", "Detection Quality"])
path='../experiments/initial data/'
patients=os.listdir(path)
