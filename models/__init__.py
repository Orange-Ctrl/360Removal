import os
import sys
from .models import register, make
from . import gan, modules, coordfill, ffc_baseline
from . import misc
# 将TMM目录的父目录添加到sys.path中
script_dir = os.path.dirname(__file__)  # 获取当前脚本的目录
project_dir = os.path.dirname(script_dir)  # 获取项目根目录
tmm_dir = os.path.join(project_dir, 'TMM')  # 获取TMM目录的路径
if tmm_dir not in sys.path:
    sys.path.append(tmm_dir)

from networks import unifuse
from networks import unifuse_weight

from TMM.losses import SVS_plane