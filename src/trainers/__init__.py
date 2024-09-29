""" Training package """

from trainers.xla_hlm_trainer import XLAHLmTrainer
from trainers.xla_zlm_trainer import XLAZLmTrainer
from trainers.xla_markovlm_trainer import XLAMarkovLmTrainer

TRAINER_DICT = {
    "XLAHLmTrainer": XLAHLmTrainer,
    "XLAZLmTrainer": XLAZLmTrainer,
    "XLAMarkovLmTrainer": XLAMarkovLmTrainer
}
