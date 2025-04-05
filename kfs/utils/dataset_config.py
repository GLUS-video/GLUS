import os.path as osp



# RVOS 
RVOS_ROOT = "PATH_TO_DATA"
RVOS_DATA_INFO = {
    "mevis_train"   : ("mevis/train",           "mevis/train/meta_expressions.json"),
    "mevis_val"     : ("mevis/valid_u",         "mevis/valid_u/meta_expressions.json"),
    "mevis_test"    : ("mevis/valid",           "mevis/valid/meta_expressions.json"),
    "refytvos_train": ('Refer-YouTube-VOS/train', 'Refer-YouTube-VOS/meta_expressions/train/meta_expressions.json'),
    "refytvos_valid": ('Refer-YouTube-VOS/valid', 'Refer-YouTube-VOS/meta_expressions/valid/meta_expressions.json'),
}