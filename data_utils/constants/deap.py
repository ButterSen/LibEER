from ..preprocess import generate_adjacency_matrix, generate_rgnn_adjacency_matrix
from .channel_location import system_10_05_loc
DEAP_CHANNEL_NAME = ['FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
                'OZ', 'PZ', 'FP2', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC2', 'CZ', 'C4', 'T8', 'CP6', 'CP2',
                'P4', 'P8', 'PO4', 'O2']

HSLT_DEAP_Regions = {
    'PF': ['FP1', 'AF3', 'AF4', 'FP2'],
    'F': ['F7', 'F3', 'FZ', 'F4', 'F8'],
    'LT': ['FC5', 'T7', 'CP5'],
    'C': ['FC1', 'C3', 'CZ', 'C4', 'FC2'],
    'RT': ['FC6', 'T8', 'CP6'],
    'LP': ['P7', 'P3', 'PO3'],
    'P': ['CP1', 'PZ', 'CP2'],
    'RP': ['P8', 'P4', 'PO4'],
    'O': ['O1', 'OZ', 'O2']
}
DEAP_GLOBAL_CHANNEL_PAIRS = [
    ['FP1', 'FP2'],
    ['AF3', 'AF4'],
    ['FC5', 'FC6'],
    ['CP5', 'CP6'],
    ['O1', 'O2']
]

DEAP_RGNN_ADJACENCY_MATRIX = generate_rgnn_adjacency_matrix(channel_names=DEAP_CHANNEL_NAME, channel_loc=system_10_05_loc,global_channel_pair=DEAP_GLOBAL_CHANNEL_PAIRS)




