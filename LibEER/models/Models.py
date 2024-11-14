from models.DGCNN import DGCNN
# from models.RGNN import RGNN
from models.RGNN_official import SymSimGCNNet
from models.EEGNet import EEGNet
from models.STRNN import STRNN
from models.GCBNet import GCBNet
from models.DBN import DBN
from models.TSception import TSception
from models.SVM import SVM
from models.CDCN import CDCN
from models.HSLT import HSLT
from models.ACRNN import ACRNN
from models.GCBNet_BLS import GCBNet_BLS
from models.MsMda import MSMDA
from models.R2GSTNN import R2GSTNN
from models.BiDANN import BiDANN

Model = {
    'DGCNN': DGCNN,
    'R2GSTNN': R2GSTNN,
    'BiDANN': BiDANN,
    'RGNN_official': SymSimGCNNet,
    'GCBNet': GCBNet,
    'GCBNet_BLS': GCBNet_BLS,
    'CDCN': CDCN,
    'DBN': DBN,
    'STRNN': STRNN,
    'EEGNet': EEGNet,
    'HSLT': HSLT,
    'ACRNN': ACRNN,
    'TSception': TSception,
    'MsMda': MSMDA,
    'svm' : SVM,
}
