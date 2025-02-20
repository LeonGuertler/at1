import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import textarena as ta
from wrappers import AnswerTokenAgentWrapper, OpenRouterAgent
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import concurrent.futures
from tqdm import tqdm
import numpy as np
import torch
from typing import List, Optional, Dict
from accelerate import Accelerator

# Constants
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ENVIRONMENT_ID = "ConnectFour-v0"
NUM_PARALLEL_ENVS = 8

EPSILON = 0.2
GAMMA = 0.8
LR = 2e-5
KL_COEFF = 0.2
EPOCHS = 3
BATCH_SIZE = 8
MAX_GRADIENT_NORM = 1.0
MAX_NEW_TOKENS = 8192
OUTPUT_DIR = "./connect_four_model"
BIG_MODEL_NAMES = None

STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."
    
class ConnectFourPPOTrainer:
    def __init__(self, model_path=MODEL_PATH, load_in_4bit=True, lora_r=16):
        """Initialize the PPO trainer with a DeepSeek R1 model and LoRA"""
        self.system_prompt = STANDARD_GAME_PROMPT
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.accelerator = Accelerator()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with quantization
        print(f"Loading model {model_path} on {self.device}...")
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Initialize LoRA 
        self.setup_lora(r=lora_r)
        self.optimizer = AdamW(self.model.parameters(), lr=LR, weight_decay=0.01)
        self.global_step = 0

        # Prepare model and optimizer with Accelerate (this moves them to the correct device(s))
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        
    def setup_lora(self, r=16):
        """Set up Low-Rank Adaptation for parameter efficient fine-tuning"""
        if hasattr(self.model, "is_loaded_in_4bit") and self.model.is_loaded_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
            
        config = LoraConfig(
            r=r,
            lora_alpha=2*r,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

    def __call__(self, user_input: str) -> str:
        # 1) Combine system prompt + user input
        prompt = self.system_prompt + "\n" + user_input

        # 2) Tokenize the entire prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs["input_ids"].shape[1]

        # 3) Generate
        generation_output = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
        )

        # 4) Slice out only the newly generated tokens
        #    i.e., everything after the prompt_length.
        new_tokens = generation_output[0][prompt_length:]

        # 5) Decode only the new tokens
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # 6) (Optional) Trim spaces
        output_text = output_text.strip()

        return output_text



    def batch_generate(self, user_inputs: List[str]) -> List[str]:
        """
        Perform a single forward pass on a batch of inputs.
        Returns a list of raw decoded strings (one per input).
        """
        # 1) Build the "prompt" for each user input
        #    E.g. prepend self.system_prompt to each user input
        batch_prompts = [self.system_prompt + "\n" + ui for ui in user_inputs]

        # 2) Tokenize as a batch
        encodings = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # Keep track of each prompt length so we can slice out only new tokens later
        prompt_lengths = []
        for i in range(encodings["input_ids"].shape[0]):
            # Number of tokens in row i
            prompt_len_i = (encodings["input_ids"][i] != self.tokenizer.pad_token_id).sum()
            prompt_lengths.append(prompt_len_i.item())

        # 3) Generate all at once
        outputs = self.model.generate(
            **encodings,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2
        )

        # 4) Decode each row’s newly generated tokens
        decoded_results = []
        for i in range(outputs.shape[0]):
            # Slice out the “new” tokens (everything after the prompt)
            new_tokens = outputs[i][prompt_lengths[i]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            decoded_results.append(text)

        return decoded_results


    def collect_experience(
        self,
        n_episodes=100,
        epsilon=0.1,
        num_envs=8,
        big_model_names=None
    ):
        """
        Collect self-play experience with multiple parallel environments.
        - Local model actions are done in a single batch pass per round.
        - Big-model (OpenRouter) calls are done in parallel threads.
        """

        if big_model_names is None:
            big_model_names = ["deepseek/deepseek-r1"]

        # 1) Wrap *this* trainer in an AnswerTokenAgentWrapper for local usage:
        local_agent = AnswerTokenAgentWrapper(self, answer_token="### Final Answer")

        # 2) We'll create one "template" OpenRouterAgent wrapper. In practice,
        #    if you want to dynamically choose different big_model_names, you can
        #    create them on the fly, or pick randomly each time. Example below.
        #    Or you can re-instantiate an agent for each big_model_name, etc.
        #    We'll keep it simple by storing no model_name at init, and picking
        #    each time in parallel calls. 
        #    (Alternatively, define separate wrappers or pass model_name in the call.)
        def openrouter_inference(observation, chosen_name):
            # Build an ephemeral agent each time
            big_agent = AnswerTokenAgentWrapper(OpenRouterAgent(model_name=chosen_name))
            return big_agent.call_with_full_answer(observation)

        # 3) Create parallel envs
        envs = []
        for _ in range(num_envs):
            env = ta.make(ENVIRONMENT_ID)
            env = ta.wrappers.LLMObservationWrapper(env=env)
            env.reset(num_players=2)
            envs.append(env)

        dones = [False] * num_envs
        episode_observations = [{0: [], 1: []} for _ in range(num_envs)]
        episode_actions      = [{0: [], 1: []} for _ in range(num_envs)]
        turns_count = [0]*num_envs

        completed_episodes = 0
        full_observations, full_actions, full_rewards = [], [], []

        stats = {
            "model_wins": 0,
            "opponent_wins": 0,
            "draws": 0,
            "avg_turns": 0
        }

        pbar = tqdm(total=n_episodes, desc="Collecting experiences")

        # MAIN LOOP
        while completed_episodes < n_episodes:
            # 1) find all active envs
            active_indices = [i for i, d in enumerate(dones) if not d]
            if not active_indices:
                # all done
                break

            # gather local vs big-model requests
            local_requests = []         # (env_idx, player_id, observation)
            openrouter_requests = []    # (env_idx, player_id, observation, chosen_name)

            for i in active_indices:
                player_id, obs = envs[i].get_observation()
                # Epsilon-greedy choice: local vs big
                if np.random.rand() < epsilon:
                    # pick a random big model from the list
                    chosen_name = np.random.choice(big_model_names)
                    openrouter_requests.append((i, player_id, obs, chosen_name))
                else:
                    local_requests.append((i, player_id, obs))

            # 2) Single batch call for local requests (if any)
            local_actions = []
            if local_requests:
                # Extract just the observation strings
                local_obs_list = [r[2] for r in local_requests]
                # One pass to get all answers
                local_results = local_agent.batch_call_with_full_answer(local_obs_list)
                # local_results is a list of (raw_answer, parsed_answer), same length

                # We then pair them back to the environment indices
                for (env_idx, pid, obs), (raw_answer, parsed_answer) in zip(local_requests, local_results):
                    local_actions.append((env_idx, pid, obs, raw_answer, parsed_answer))

            # 3) Parallel calls for openrouter requests (if any)
            openrouter_actions = []
            if openrouter_requests:
                # We'll run them in parallel
                def do_openrouter(req):
                    # req is (env_idx, pid, obs, chosen_name)
                    env_i, p_id, observation, model_name = req
                    raw_answer, parsed_answer = openrouter_inference(observation, model_name)
                    return (env_i, p_id, observation, raw_answer, parsed_answer)

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(do_openrouter, req) for req in openrouter_requests]
                    results = [f.result() for f in concurrent.futures.as_completed(futures)]
                    openrouter_actions.extend(results)

            # Combine both sets of actions so we can step the envs in one pass
            all_actions_this_round = local_actions + openrouter_actions
            # We'll store them in a dict by env_idx so we know which one belongs where
            actions_by_env = {}
            for (env_idx, pid, obs, raw_answer, parsed_answer) in all_actions_this_round:
                actions_by_env[env_idx] = (pid, obs, raw_answer, parsed_answer)

            # 4) Step each active env with the chosen action
            for i in active_indices:
                # If no action was chosen for this env (possible if lists are empty?), skip
                if i not in actions_by_env:
                    continue

                (player_id, obs, raw_action, parsed_action) = actions_by_env[i]

                # store transitions
                episode_observations[i][player_id].append(obs)
                episode_actions[i][player_id].append(raw_action)

                # step environment
                done, info = envs[i].step(action=parsed_action)
                turns_count[i] += 1

                if done:
                    dones[i] = True
                    rewards = envs[i].close()

                    # Determine winner
                    if rewards[0] == 1:
                        stats["model_wins"] += 1
                        winner = 0
                    elif rewards[1] == 1:
                        stats["model_wins"] += 1  # or stats["opponent_wins"] += 1
                        winner = 1
                    else:
                        stats["draws"] += 1
                        winner = None

                    # store transitions if there's a winner
                    if winner is not None:
                        for pid in [0, 1]:
                            T = len(episode_actions[i][pid])
                            if T == 0:
                                continue
                            # discount
                            time_weights = np.array([
                                (1 - GAMMA) * (GAMMA ** (T - t - 1)) for t in range(T)
                            ])
                            time_weights /= (1 - (GAMMA ** T))

                            discounted_rewards = rewards[pid] * time_weights

                            full_observations.extend(episode_observations[i][pid])
                            full_actions.extend(episode_actions[i][pid])
                            full_rewards.extend(discounted_rewards)

                    completed_episodes += 1
                    stats["avg_turns"] = ((stats["avg_turns"] * (completed_episodes - 1))
                                        + turns_count[i]) / completed_episodes
                    pbar.update(1)

                    # If you want to gather *more* episodes from each env, you can reset here.
                    # For example:
                    # envs[i].reset(num_players=2)
                    # dones[i] = False
                    # But typically for a simple scenario, we just keep it done.

        pbar.close()

        # Stats summary
        print(f"\nExperience collection complete:")
        print(f"  Episodes: {completed_episodes}")
        print(f"  Model win rate: {stats['model_wins']/completed_episodes:.2f}")
        print(f"  Draw rate: {stats['draws']/completed_episodes:.2f}")
        print(f"  Average turns per game: {stats['avg_turns']:.1f}")
        print(f"  Total samples collected: {len(full_rewards)}")

        return full_observations, full_actions, full_rewards, stats


    
    def prepare_training_batch(self, observations, actions, rewards):
        """Prepare input data for PPO training"""
        dataset = [
            {
                "input": obs,
                "output": act,
                "reward": rew 
            }
            for obs, act, rew in zip(observations, actions, rewards)
        ]
        
        # Filter to include only examples with positive rewards
        positive_examples = [ex for ex in dataset if ex["reward"] > 0]
        return positive_examples
    
    def compute_kl_loss(self, q_logits, p_logits):
        """Compute KL divergence loss between current and reference policy"""
        p_log_softmax = torch.nn.functional.log_softmax(p_logits, dim=-1)
        q_softmax = torch.nn.functional.softmax(q_logits, dim=-1)
        kl = torch.sum(q_softmax * (torch.log(q_softmax) - p_log_softmax), dim=-1)
        return kl.mean()
        
    def train_ppo_step(self, batch):
        """Train a single PPO step on the collected experience"""
        self.model.train()
        
        # Create a copy of the model for KL penalty
        ref_model = None
        if KL_COEFF > 0:
            ref_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True
            )
            ref_model.eval()
            
        # Process each example in the batch
        total_loss = 0
        
        for example in batch:
            # Format as instruction
            input_text = example["input"]
            target_text = example["output"]
            reward = example["reward"]
            
            # Tokenize
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            targets = self.tokenizer(target_text, return_tensors="pt").to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs, labels=targets.input_ids)
            loss = outputs.loss
            
            # Add KL penalty if using reference model
            if ref_model is not None:
                with torch.no_grad():
                    ref_outputs = ref_model(**inputs)
                kl_loss = self.compute_kl_loss(outputs.logits, ref_outputs.logits)
                loss = loss + KL_COEFF * kl_loss
                
            # Apply reward weighting
            loss = loss * (1.0 - reward)  # Lower loss for higher rewards
            
            total_loss += loss
            
        # Average loss over batch
        avg_loss = total_loss / len(batch)
        
        # Backward and optimize
        self.optimizer.zero_grad()
        # Use Accelerate's backward to support distributed training
        self.accelerator.backward(avg_loss)
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRADIENT_NORM)
        self.optimizer.step()
        
        return avg_loss.item()
    
    # def save_checkpoint(self, output_dir=OUTPUT_DIR):
    #     """Save model checkpoint"""
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
            
    #     checkpoint_dir = os.path.join(output_dir, f"checkpoint-{self.global_step}")
    #     self.model.save_pretrained(checkpoint_dir)
    #     self.tokenizer.save_pretrained(checkpoint_dir)
        
    #     print(f"Model saved to {checkpoint_dir}")
    #     return checkpoint_dir
    def save_checkpoint(self, output_dir=OUTPUT_DIR):
        """Save model checkpoint (only on the main process)"""
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{self.global_step}")
            # Unwrap model to get the original model for saving
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            print(f"Model saved to {checkpoint_dir}")
            return checkpoint_dir
    
    def evaluate(self, n_games=50):
        """Evaluate the model by playing against a random agent"""
        random_agent = ta.agents.RandomAgent()
        model_agent = AnswerTokenAgentWrapper(self)
        
        wins, losses, draws = 0, 0, 0
        
        for _ in tqdm(range(n_games), desc="Evaluating"):
            # Randomly assign player order
            if np.random.uniform() < 0.5:
                agents = {0: model_agent, 1: random_agent}
                model_player = 0
            else:
                agents = {0: random_agent, 1: model_agent}
                model_player = 1
                
            # Build environment
            env = ta.make(ENVIRONMENT_ID)
            env = ta.wrappers.LLMObservationWrapper(env=env)
            
            # Run game
            env.reset()
            done = False
            while not done:
                player_id, observation = env.get_observation()
                action = agents[player_id](observation)
                done, _ = env.step(action=action)
                
            rewards = env.close()
            
            if rewards[model_player] == 1:
                wins += 1
            elif rewards[model_player] == -1:
                losses += 1
            else:
                draws += 1
                
        win_rate = wins / n_games
        
        results = {
            "win_rate": win_rate,
            "draw_rate": draws / n_games,
            "loss_rate": losses / n_games,
            "games_played": n_games
        }
        
        print(f"Evaluation results:")
        print(f"  Win rate: {win_rate:.2f}")
        print(f"  Draw rate: {results['draw_rate']:.2f}")
        print(f"  Loss rate: {results['loss_rate']:.2f}")
        
        return results
    
    def train(self, n_epochs=EPOCHS, episodes_per_epoch=100, batch_size=BATCH_SIZE, save_freq=1, evaluate_freq=1):
        """Run full PPO training loop"""
        print(f"Starting PPO training for {n_epochs} epochs")
        
        for epoch in range(n_epochs):
            print(f"\n--- Epoch {epoch+1}/{n_epochs} ---")

 
            # 1. Collect experience
            observations, actions, rewards, stats = self.collect_experience(
                n_episodes=episodes_per_epoch,
                epsilon=EPSILON,
                num_envs=NUM_PARALLEL_ENVS,
                big_model_names=BIG_MODEL_NAMES
            )
            
            # 2. Prepare training data
            training_data = self.prepare_training_batch(observations, actions, rewards)
            print(f"Training on {len(training_data)} positive examples")
            
            # 3. Train model
            epoch_losses = []
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                loss = self.train_ppo_step(batch)
                epoch_losses.append(loss)
                self.global_step += 1
                
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
            
            # 4. Evaluation
            if evaluate_freq > 0 and (epoch + 1) % evaluate_freq == 0:
                eval_results = self.evaluate(n_games=50)
                
            # 5. Save checkpoint
            if save_freq > 0 and (epoch + 1) % save_freq == 0:
                self.save_checkpoint()
                
        # # Final save
        # final_dir = os.path.join(OUTPUT_DIR, "final_model")
        # self.model.save_pretrained(final_dir)
        # self.tokenizer.save_pretrained(final_dir)
        # print(f"Training complete! Final model saved to {final_dir}")
        
        # # Final evaluation
        # final_results = self.evaluate(n_games=100)
        # return final_results
        # Final save (only on main process)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            final_dir = os.path.join(OUTPUT_DIR, "final_model")
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(final_dir)
            self.tokenizer.save_pretrained(final_dir)
            print(f"Training complete! Final model saved to {final_dir}")
            
        final_results = self.evaluate(n_games=100)
        return final_results

# Main training function
def train_connect_four_model():
    # Initialize trainer
    trainer = ConnectFourPPOTrainer(
        model_path=MODEL_PATH,
        load_in_4bit=True,
        lora_r=16
    )
    
    # Run training
    results = trainer.train(
        n_epochs=3,
        episodes_per_epoch=100,
        batch_size=8,
        save_freq=1,
        evaluate_freq=1
    )
    
    return trainer, results

if __name__ == "__main__":
    trainer, results = train_connect_four_model()