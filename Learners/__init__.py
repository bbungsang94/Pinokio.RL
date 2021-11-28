from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .LIIR_Learner import LIIRLearner
from .latent_q_learner import LatentQLearner
from .rode_learner import RODELearner
from .Central_V_Learner import CentralV_Learner
from .noise_q_learner import NoiseQLearner
from .offpg_learner import OffPGLearner
from .rode_gat_learner import RODE_GAT_Learner

REGISTRY = {"q_learner": QLearner, "coma_learner": COMALearner, "qtran_learner": QTranLearner,
            "liir_learner": LIIRLearner, 'latent_q_learner': LatentQLearner, 'rode_learner': RODELearner,
            'centralV': CentralV_Learner, "noise_q_learner": NoiseQLearner, "offpg_learner": OffPGLearner,
            'rode_gat_learner': RODE_GAT_Learner}

