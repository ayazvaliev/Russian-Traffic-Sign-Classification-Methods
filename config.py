__all__ = ['NET_NAME',
           'NET_SIZE',
           'UNFREEZED_LAYERS',
           'CROP_SIZE', 
           'MEAN', 
           'STD', 
           'SMOOTHING_RATIO',
           'WARMUP_EPOCHS',
           'MAX_EPOCHS', 
           'BASE_LR', 
           'BATCH_SIZE', 
           'ELEMS_PER_CLASS',
           'CLASSES_PER_BATCH',
           'N_NEIGHBORS',
           'EXAMPLES_PER_CLASS',
           'INDEX_PER_CLASS',
           'N_ESTIMATORS'
           'INTERNAL_FEATURES',
           'MERGED_LOSS_COEF',
           'MARGIN',
           'SYNT_ONLY',
           'DEVICE'
           ]


SYNT_ONLY = None
MARGIN = None
MERGED_LOSS_COEF = None
INTERNAL_FEATURES = None
NET_NAME = None
UNFREEZED_LAYERS = None
NET_SIZE = None
CROP_SIZE = None
MEAN = None
STD = None
ELEMS_PER_CLASS = None
CLASSES_PER_BATCH = None
SMOOTHING_RATIO = None
WARMUP_EPOCHS = None
MAX_EPOCHS = None
BASE_LR = None
BATCH_SIZE = None
N_NEIGHBORS = None
INDEX_PER_CLASS = None
N_ESTIMATORS = None


# Constants along all experiments

CLASSES_CNT = 205
DEVICE = 'cpu'
TRAIN_DATASET_PATH = 'resources/cropped-train'
SYNT_DATASET_PATH = 'resources/synt-data'
TEST_DATASET_PATH = 'resources/smalltest'
CLASSES_INFO_PATH = 'resources/classes.json'
TEST_ANNOTATIONS_PATH = 'resources/signs/smalltest_annotations.csv'
CKPTS_PATH = 'checkpoints/'

