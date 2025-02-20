""" General config """


MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ENVIRONMENT_ID = "SpellingBee-v0" #"ConnectFour-v0"
NUM_PARALLEL_ENVS = 10

EPSILON = 1.0
GAMMA = 0.8
LR = 2e-5
KL_COEFF = 0.2

EPOCHS = 3
BATCH_SIZE = 8
MAX_GRADIENT_NORM = 1.0
MAX_NEW_TOKENS = 2048*2 #8192

OUTPUT_DIR = "./connect_four_model"
BIG_MODEL_NAMES = ["deepseek/deepseek-r1-distill-qwen-32b"]
# BIG_MODEL_NAMES = ["deepseek/deepseek-r1-distill-qwen-1.5b"]

STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."
