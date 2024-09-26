""" Training package """

from trainers.xla_lm_trainer import XLALMTrainer
from trainers.xla_swift_trainer import XLASwiftTrainer
from trainers.xla_vaelm_trainer import XLAVaeLmTrainer
from trainers.xla_hlm_trainer import XLAHLmTrainer

TRAINER_DICT = {
    "XLALMTrainer": XLALMTrainer,
    "XLASwiftTrainer": XLASwiftTrainer,
    "XLAVaeLmTrainer": XLAVaeLmTrainer,
    "XLAHLmTrainer": XLAHLmTrainer,
}
