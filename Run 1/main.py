from model import ConnectFourPPOTrainer
from config import MODEL_PATH

# Initialize trainer
trainer = ConnectFourPPOTrainer(
    model_path=MODEL_PATH,
    load_in_4bit=True,
    lora_r=16
)

# Run training
results = trainer.train(
    n_epochs=3,
    episodes_per_epoch=10, #0,
    batch_size=8,
    save_freq=1,
    evaluate_freq=1
)
