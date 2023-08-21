from pathlib import Path
from dotenv import dotenv_values
import os

# ROOT
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
config = dotenv_values(HERE / "root.env")
ROOT = Path(config['ROOT'])
SIDE_DATA = Path('/content/drive/MyDrive/OCR/remittance_layoutlm_debug/clusters/77/output')

# PROJECT TREE
DATA = ROOT / 'DATA'
CONFIGS = ROOT / 'configs'
CFGM = CONFIGS / 'models'
OUTPUTS = ROOT / 'outputs'
RUNS = OUTPUTS / 'runs'
RESULTS = OUTPUTS / 'results'
IMGS = OUTPUTS / 'images'
TRAIN_SAMPLES = OUTPUTS / 'train_samples'
TEST_SAMPLES = OUTPUTS / 'test_samples'
VALIDATION_SAMPLES = OUTPUTS / 'validation_samples'
INFERENCE_SAMPLES = OUTPUTS / 'inference_samples'
TRAINING = ROOT / 'src' / 'training'
MODELS = ROOT / 'src' / 'models'
CHECKPOINTS = MODELS / 'checkpoints'
INFERENCE = ROOT / 'inference'

# FUNSD
FUNSD_TRAIN = DATA / 'FUNSD' / 'training_data'
FUNSD_TEST = DATA / 'FUNSD' / 'testing_data'

# PAU
PAU_TRAIN = DATA / 'PAU' / 'train'
PAU_TEST = DATA / 'PAU' / 'test'

# REMITTANCE
REMITTANCE_TRAIN = DATA/ 'REMITTANCE' / 'train'
REMITTANCE_TEST = DATA/ 'REMITTANCE' / 'test'
REMITTANCE_VAL = DATA/ 'REMITTANCE' / 'validation'
REMITTANCE_INFERENCE = DATA/ 'REMITTANCE' / 'inference'

INF = DATA / 'INF'
