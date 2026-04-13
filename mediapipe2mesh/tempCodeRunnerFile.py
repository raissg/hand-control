rmatures import *
from inverse_kinematics.models import *
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
k2m = Keypoints2Mano(model_path=os.path.join(_SCRIPT_DIR, 'MANO_RIGHT.npz'))

## ------------------- ##
##      Visualizer     ##