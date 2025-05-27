""" Training package """

from trainers.zlm_trainer import ZLmTrainer
from trainers.uncond_zlm_trainer import UncondZLmTrainer
from trainers.zae_trainer import ZAETrainer
from trainers.zlm_full_trainer import ZLmFullTrainer
from trainers.zlm_contrast_trainer import ZLmContrastTrainer
from trainers.zlm_asym_trainer import ZLmAsymTrainer
from trainers.zlm_hybrid_trainer import ZLmHybridTrainer
from trainers.ibml_trainer import IBMLTrainer


TRAINER_DICT = {
    "zlm": ZLmTrainer,
    "uncond_zlm": UncondZLmTrainer,
    "zae": ZAETrainer,
    "zlm_full": ZLmFullTrainer,
    "zlm_contrast": ZLmContrastTrainer,
    "zlm_asym": ZLmAsymTrainer,
    "zlm_hybrid": ZLmHybridTrainer,
    "ibml": IBMLTrainer,
}
