""" Training package """

from trainers.xla_hlm_trainer import XLAHLmTrainer
from trainers.xla_zlm_trainer import XLAZLmTrainer

TRAINER_DICT = {
    "XLAHLmTrainer": XLAHLmTrainer,
    "XLAZLmTrainer": XLAZLmTrainer,
}
