import cv2
from COMPARISON import execution

def detect_sift(thrm,nfeatures=2000):
    sift=cv2.SIFT_create(nfeatures=nfeatures)
    kpts,desc=sift.detectAndCompute(thrm,None)
    return kpts,desc

def detect_orb(thrm,nfeatures=2000):
    ORB=cv2.ORB_create(nfeatures=nfeatures)
    kpts,desc=ORB.detectAndCompute(thrm,None)
    return kpts,desc

def detect_brisk(thrm):
    brisk = cv2.BRISK_create()
    kpts,desc=brisk.detectAndCompute(thrm, None)
    return kpts,desc

def detect_akaze(thrm):
    akaze = cv2.AKAZE_create()
    kpts,desc=akaze.detectAndCompute(thrm, None)
    return kpts,desc
#---------------------------------------------------------------

def feature_detector():
    FEATURE_METHODS = {
    "SIFT": detect_sift,
    "ORB": detect_orb,
    "BRISK": detect_brisk,
    "AKAZE": detect_akaze
    }
    model=None
    for name,fn in FEATURE_METHODS.items():
        execution(fn,model,name)
    