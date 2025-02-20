"""
RL training loop with memory recall
1. collect n-samples
2. process the current memory recall module
3. train using KL PPO on that data
4. go back to step 1
"""
import textarena as ta 
from wrappers import AnswerTokenAgentWrapper
import numpy as np 
from tqdm import tqdm
import time
from collections import defaultdict

ENVIRONMENT_ID = "ConnectFour-v0"
BIG_MODEL_NAME = "deepseek/deepseek-r1"

# training parameters
GAMMA = 0.8


def collect_data(model, n_samples: int, use_r1_opponent: float = 0.0):
    full_observations, full_actions, full_rewards = [], [], []
    
    # Statistics tracking
    stats = {
        "episodes_played": 0,
        "model_wins": 0,
        "opponent_wins": 0,
        "draws": 0,
        "total_turns": 0,
        "win_rate": 0.0,
        "avg_turns_per_game": 0.0,
        "episodes_used": 0,  # Count of non-draw episodes used for training
        "start_time": time.time()
    }
    
    # Create progress bar
    pbar = tqdm(total=n_samples, desc="Collecting samples")
    
    # now iterate over episodes until we have sufficient data
    while len(full_rewards) < n_samples:
        # run another episode
        episode_observations, episode_actions = {0: [], 1: []}, {0: [], 1: []} 
        turn_counter = 0

        # initialize agents
        is_r1_opponent = np.random.uniform() < use_r1_opponent
        model_0 = AnswerTokenAgentWrapper(model)
        model_1 = AnswerTokenAgentWrapper(model if not is_r1_opponent else ta.agents.OpenRouterAgent(BIG_MODEL_NAME))
        agents = {0: model_0, 1: model_1}

        # build the environment
        env = ta.make(ENVIRONMENT_ID)
        env = ta.wrappers.LLMObservationWrapper(env=env)

        # reset & run the environment
        env.reset()
        done = False 
        while not done:
            # get current pid and observation
            player_id, observation = env.get_observation()

            # get model action
            raw_action, action = agents[player_id].call_with_full_answer(observation)

            # track all relevant information
            episode_observations[player_id].append(observation)
            episode_actions[player_id].append(raw_action)

            # execute action
            done, info = env.step(action=action)
            turn_counter += 1

        # get the rewards
        rewards = env.close()
        
        # Update statistics
        stats["episodes_played"] += 1
        stats["total_turns"] += turn_counter
        
        if rewards[0] == 1:
            stats["model_wins"] += 1
        elif rewards[1] == 1:
            if is_r1_opponent:
                stats["opponent_wins"] += 1
            else:
                stats["model_wins"] += 1
        else:
            stats["draws"] += 1
            
        stats["win_rate"] = stats["model_wins"] / stats["episodes_played"]
        stats["avg_turns_per_game"] = stats["total_turns"] / stats["episodes_played"]

        # Only use episodes where there's a definitive winner (not draws)
        samples_before = len(full_rewards)
        if rewards[0] == 1 or rewards[1] == 1:
            stats["episodes_used"] += 1
            
            for player_id in range(2):
                if len(episode_actions[player_id]) > 0:
                    T = len(episode_actions[player_id])
                    
                    # Apply temporal decay weighting from SPAG paper
                    time_weights = np.array([(1-GAMMA) * GAMMA**(T-t-1) for t in range(T)])
                    # Normalize weights
                    time_weights = time_weights / (1 - GAMMA**(T))
                    
                    # Apply weights to the final reward
                    discounted_player_rewards = rewards[player_id] * time_weights
                    
                    # Extend to full tracking lists
                    full_observations.extend(episode_observations[player_id])
                    full_actions.extend(episode_actions[player_id])
                    full_rewards.extend(discounted_player_rewards)
        
        # Update progress bar
        samples_added = len(full_rewards) - samples_before
        if samples_added > 0:
            pbar.update(samples_added)
            
        # Update status in progress bar description every 5 episodes
        if stats["episodes_played"] % 5 == 0:
            elapsed_time = time.time() - stats["start_time"]
            pbar.set_description(
                f"Collecting samples | Win rate: {stats['win_rate']:.2f} | "
                f"Draws: {stats['draws']/stats['episodes_played']:.2f} | "
                f"Avg turns: {stats['avg_turns_per_game']:.1f} | "
                f"Used: {stats['episodes_used']}/{stats['episodes_played']}"
            )

    pbar.close()
    
    # Print final statistics
    print(f"\nData collection complete:")
    print(f"  Episodes played: {stats['episodes_played']}")
    print(f"  Episodes used for training: {stats['episodes_used']} ({stats['episodes_used']/stats['episodes_played']*100:.1f}%)")
    print(f"  Model win rate: {stats['win_rate']:.2f}")
    print(f"  Draw rate: {stats['draws']/stats['episodes_played']:.2f}")
    print(f"  Average turns per game: {stats['avg_turns_per_game']:.1f}")
    if use_r1_opponent > 0:
        opponent_games = int(stats['episodes_played'] * use_r1_opponent)
        if opponent_games > 0:
            expert_win_rate = stats['opponent_wins'] / opponent_games
            print(f"  Expert opponent win rate: {expert_win_rate:.2f} (in {opponent_games} games)")
    print(f"  Total samples collected: {len(full_rewards)}")
    
    return full_observations, full_actions, full_rewards, stats


# test data collection
full_observations, full_actions, full_rewards, stats = collect_data(model=)