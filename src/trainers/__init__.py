""" Training package """

from trainers.zlm_trainer import ZLmTrainer
from trainers.uncond_zlm_trainer import UncondZLmTrainer

TRAINER_DICT = {
    "zlm": ZLmTrainer,
    "uncond_zlm": UncondZLmTrainer,
}
