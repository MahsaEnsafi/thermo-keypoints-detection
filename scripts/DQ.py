from scripts.classical.CLASIC_FD import feature_detector as CLASIC_DQ

from scripts.deep.DL_FD import feature_detector as DL_DQ

if __name__ =='__main__':
    # quality detection using classic methods for key points detection
    CLASIC_DQ()
    # quality detection using deep learning based methods for key points detection
    DL_DQ()



