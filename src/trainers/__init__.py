""" Training package """

from trainers.xla_lm_trainer import XLALMTrainer
from trainers.xla_swift_trainer import XLASwiftTrainer

TRAINER_DICT = {
    "XLALMTrainer": XLALMTrainer,
    "XLASwiftTrainer": XLASwiftTrainer,
}
