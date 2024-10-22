# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN

def add_detic_config(cfg):
    _C = cfg

    # Open-vocabulary classifier
    _C.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS = False # Use fixed classifier for open-vocabulary detection
    _C.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'datasets/metadata/lvis_v1_clip_a+cname.npy'
    _C.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM = 512
    _C.MODEL.ROI_BOX_HEAD.NORM_WEIGHT = True
    # _C.MODEL.ROI_BOX_HEAD.NORM_TEMP = 25.0
    _C.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS = False
    # _C.MODEL.ROI_BOX_HEAD.USE_BIAS = 0.0 # >= 0: not use
    _C.MODEL.ROI_BOX_HEAD.FIX_BIAS = False
    
    _C.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE = False # CenterNet2
    _C.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    _C.MODEL.ROI_BOX_HEAD.PRIOR_PROB = 0.01
    _C.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False # Federated Loss
    _C.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = \
        'datasets/metadata/lvis_v1_train_cat_info.json'
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT = 50
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT = 0.5

    # Classification data configs
    _C.MODEL.ROI_BOX_HEAD.IMAGE_LABEL_LOSS = 'max_size' # max, softmax, sum
    _C.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE = 1.0
    _C.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX = False # Used for image-box loss and caption loss
    _C.MODEL.ROI_BOX_HEAD.WITH_SOFTMAX_PROP = False # Used for WSDDN
    _C.MODEL.ROI_BOX_HEAD.CAPTION_WEIGHT = 1.0 # Caption loss weight
    _C.MODEL.ROI_BOX_HEAD.NEG_CAP_WEIGHT = 0.125 # Caption loss hyper-parameter
    _C.MODEL.ROI_BOX_HEAD.ADD_FEATURE_TO_PROP = False # Used for WSDDN
    _C.MODEL.ROI_BOX_HEAD.SOFTMAX_WEAK_LOSS = False # Used when USE_SIGMOID_CE is False

    _C.MODEL.ROI_HEADS.MASK_WEIGHT = 1.0
    _C.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = False # For demo only

    # Caption losses
    _C.MODEL.CAP_BATCH_RATIO = 4 # Ratio between detection data and caption data
    _C.MODEL.WITH_CAPTION = False
    _C.MODEL.SYNC_CAPTION_BATCH = False # synchronize across GPUs to enlarge # "classes"

    # dynamic class sampling when training with 21K classes
    _C.MODEL.DYNAMIC_CLASSIFIER = False
    _C.MODEL.NUM_SAMPLE_CATS = 50

    # Different classifiers in testing, used in cross-dataset evaluation
    _C.MODEL.RESET_CLS_TESTS = False
    _C.MODEL.TEST_CLASSIFIERS = []
    _C.MODEL.TEST_NUM_CLASSES = []

    # Backbones
    _C.MODEL.SWIN = CN()
    _C.MODEL.SWIN.SIZE = 'T' # 'T', 'S', 'B'
    _C.MODEL.SWIN.USE_CHECKPOINT = False
    _C.MODEL.SWIN.OUT_FEATURES = (1, 2, 3) # FPN stride 8 - 32

    _C.MODEL.TIMM = CN()
    _C.MODEL.TIMM.BASE_NAME = 'resnet50'
    _C.MODEL.TIMM.OUT_LEVELS = (3, 4, 5)
    _C.MODEL.TIMM.NORM = 'FrozenBN'
    _C.MODEL.TIMM.FREEZE_AT = 0
    _C.MODEL.TIMM.PRETRAINED = False
    _C.MODEL.DATASET_LOSS_WEIGHT = []
    
    # Multi-dataset dataloader
    _C.DATALOADER.DATASET_RATIO = [1, 1]  # sample ratio
    _C.DATALOADER.USE_RFS = [False, False]
    _C.DATALOADER.MULTI_DATASET_GROUPING = False  # Always true when multi-dataset is enabled
    _C.DATALOADER.DATASET_ANN = ['box', 'box']  # Annotation type of each dataset
    _C.DATALOADER.USE_DIFF_BS_SIZE = False  # Use different batchsize for each dataset
    _C.DATALOADER.DATASET_BS = [8, 32]  # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.DATASET_INPUT_SIZE = [896, 384]  # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.DATASET_INPUT_SCALE = [(0.1, 2.0), (0.5, 1.5)]  # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.DATASET_MIN_SIZES = [(640, 800), (320, 400)]  # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.DATASET_MAX_SIZES = [1333, 667]  # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.USE_TAR_DATASET = False  # for ImageNet-21K, directly reading from unziped files
    _C.DATALOADER.TARFILE_PATH = 'datasets/imagenet/metadata-22k/tar_files.npy'
    _C.DATALOADER.TAR_INDEX_DIR = 'datasets/imagenet/metadata-22k/tarindex_npy'
    
    _C.SOLVER.USE_CUSTOM_SOLVER = False
    _C.SOLVER.OPTIMIZER = 'SGD'
    _C.SOLVER.BACKBONE_MULTIPLIER = 1.0 # Used in DETR
    _C.SOLVER.CUSTOM_MULTIPLIER = 1.0 # Used in DETR
    _C.SOLVER.CUSTOM_MULTIPLIER_NAME = [] # Used in DETR

    # Deformable DETR
    _C.MODEL.DETR = CN()
    _C.MODEL.DETR.NUM_CLASSES = 80
    _C.MODEL.DETR.FROZEN_WEIGHTS = '' # For Segmentation
    _C.MODEL.DETR.GIOU_WEIGHT = 2.0
    _C.MODEL.DETR.L1_WEIGHT = 5.0
    _C.MODEL.DETR.DEEP_SUPERVISION = True
    _C.MODEL.DETR.NO_OBJECT_WEIGHT = 0.1
    _C.MODEL.DETR.CLS_WEIGHT = 2.0
    _C.MODEL.DETR.NUM_FEATURE_LEVELS = 4
    _C.MODEL.DETR.TWO_STAGE = False
    _C.MODEL.DETR.WITH_BOX_REFINE = False
    _C.MODEL.DETR.FOCAL_ALPHA = 0.25
    _C.MODEL.DETR.NHEADS = 8
    _C.MODEL.DETR.DROPOUT = 0.1
    _C.MODEL.DETR.DIM_FEEDFORWARD = 2048
    _C.MODEL.DETR.ENC_LAYERS = 6
    _C.MODEL.DETR.DEC_LAYERS = 6
    _C.MODEL.DETR.PRE_NORM = False
    _C.MODEL.DETR.HIDDEN_DIM = 256
    _C.MODEL.DETR.NUM_OBJECT_QUERIES = 100

    _C.MODEL.DETR.USE_FED_LOSS = False
    _C.MODEL.DETR.WEAK_WEIGHT = 0.1

    _C.INPUT.CUSTOM_AUG = ''
    _C.INPUT.TRAIN_SIZE = 640
    _C.INPUT.TEST_SIZE = 640
    _C.INPUT.SCALE_RANGE = (0.1, 2.)
    # 'default' for fixed short/ long edge, 'square' for max size=INPUT.SIZE
    _C.INPUT.TEST_INPUT_TYPE = 'default' 

    _C.FIND_UNUSED_PARAM = True
    _C.EVAL_PRED_AR = False
    _C.EVAL_PROPOSAL_AR = False
    _C.EVAL_CAT_SPEC_AR = False
    _C.IS_DEBUG = False
    _C.QUICK_DEBUG = False
    _C.FP16 = False
    _C.EVAL_AP_FIX = False
    _C.GEN_PSEDO_LABELS = False
    _C.SAVE_DEBUG_PATH = 'output/save_debug/'

    # customize
    # clip
    _C.MODEL.CLIP = CN()
    _C.MODEL.CLIP.NAME = 'ViT-B/32'
    _C.MODEL.CLIP.USE_IMAGE_ENCODER = False
    _C.MODEL.CLIP.MODEL_ROOT = 'models'
    _C.MODEL.CLIP.INPUT_RESOLUTION = 224

    # _C.MODEL.ROI_BOX_HEAD.PAD_FIRST = True
    # _C.MODEL.ROI_BOX_HEAD.ALL_ENCODER = False
    _C.MODEL.ROI_BOX_HEAD.RANDOM_DROPOUT = 0.5
    _C.MODEL.ROI_BOX_HEAD.NORM_TEMP = 25.0
    _C.MODEL.ROI_BOX_HEAD.USE_BIAS = -20.0  # >= 0: not use
    _C.MODEL.ROI_BOX_HEAD.NUM_WORDS = 4
    _C.MODEL.ROI_BOX_HEAD.WORD_EMBED_DIM = 512
    _C.MODEL.ROI_BOX_HEAD.CLS_LOSS_WEIGHT = 1.0
    _C.MODEL.ROI_BOX_HEAD.COSINE_SCORE = False
    _C.MODEL.ROI_BOX_HEAD.NOVEL_BIAS = 0.0
    _C.MODEL.ROI_BOX_HEAD.NOVEL_TEMP = 1.0
    _C.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS = 32  # num proposals for image-labeled data
    _C.MODEL.ROI_BOX_HEAD.IMAGE_POS_WEIGHT = 1.0
    _C.MODEL.ROI_BOX_HEAD.IMAGE_LOSS_WEIGHT = 1.0
    _C.MODEL.WITH_IMAGE_LABELS = False  # Turn on co-training with classification data

    # context sampling
    _C.CONTEXT_MODELLING = CN()
    _C.CONTEXT_MODELLING.ENABLE = False

    _C.CONTEXT_MODELLING.CONTRAST_LOSS_WEIGHT = 1.0
    _C.CONTEXT_MODELLING.TOKEN_LOSS_WEIGHT = 0.1
    _C.CONTEXT_MODELLING.CAPTION_LOSS_WEIGHT = 2.0

    _C.CONTEXT_MODELLING.TOPK = 300
    _C.CONTEXT_MODELLING.NMS_THR = 0.10
    _C.CONTEXT_MODELLING.OBJECTNESS_THR = 0.85
    _C.CONTEXT_MODELLING.AREA_RATIO_THR = 0.01
    _C.CONTEXT_MODELLING.SHAPE_RATIO_THR = 0.25
    _C.CONTEXT_MODELLING.POSITIONAL_ENCODING = True
    _C.CONTEXT_MODELLING.INPUT_RESOLUTION = 224

    _C.CONTEXT_MODELLING.CE_TEMP = 30.0
    _C.CONTEXT_MODELLING.BCE_TEMP = 30.0
    _C.CONTEXT_MODELLING.TOKEN_TEMP = 50.0
    _C.CONTEXT_MODELLING.BCE_POS_WEIGHT = 10.0

    _C.CONTEXT_MODELLING.CHECKBOARD = CN()
    _C.CONTEXT_MODELLING.CHECKBOARD.ENABLE = False
    _C.CONTEXT_MODELLING.CHECKBOARD.NMS_THR = 0.10
    _C.CONTEXT_MODELLING.CHECKBOARD.AREA_RATIO_THR = 0.01
    _C.CONTEXT_MODELLING.CHECKBOARD.BASE_PROBABILITY = 0.3
    _C.CONTEXT_MODELLING.CHECKBOARD.MAX_GROUPS = 4
    _C.CONTEXT_MODELLING.CHECKBOARD.MAX_PERMUTATIONS = 2
    _C.CONTEXT_MODELLING.CHECKBOARD.ALPHA = 3.0
    _C.CONTEXT_MODELLING.CHECKBOARD.CUT_OFF_THR = 0.3
    _C.CONTEXT_MODELLING.CHECKBOARD.INTERVAL = 0.0
    _C.CONTEXT_MODELLING.CHECKBOARD.LOCAL_CORRESPONDENCE = True

    _C.CONTEXT_MODELLING.CAPTION = CN()
    _C.CONTEXT_MODELLING.CAPTION.ENABLE = False
    _C.CONTEXT_MODELLING.CAPTION.NMS_THR = 0.05
    _C.CONTEXT_MODELLING.CAPTION.CAPS_PER_IMG = 5
    _C.CONTEXT_MODELLING.CAPTION.AREA_RATIO_THR = 0.1
    _C.CONTEXT_MODELLING.CAPTION.MAX_NUM = 10
    _C.CONTEXT_MODELLING.CAPTION.ADD_IMAGE_BOX = True
    _C.CONTEXT_MODELLING.CAPTION.MAX_PERMUTATIONS = 4

    _C.CONTEXT_MODELLING.QUEUE = CN()
    _C.CONTEXT_MODELLING.QUEUE.LENGTHS = [1024] * 5
    _C.CONTEXT_MODELLING.QUEUE.MULTI_GPU_SYNC = False
    _C.CONTEXT_MODELLING.QUEUE.NAMES = ['clip_text_features', 'clip_image_features',
                                        'clip_word_features', 'clip_patch_features',
                                        'clip_caption_features']
