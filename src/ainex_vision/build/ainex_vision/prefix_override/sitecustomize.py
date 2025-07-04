import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/bilhr2025/Downloads/template_tB/ainex_bilhr_ws/src/ainex_vision/install/ainex_vision'
