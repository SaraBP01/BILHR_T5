import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/bilhr2025/Downloads/BILHR_T5/install/ainex_motion'
