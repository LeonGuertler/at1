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

ENVIRONMENT_ID = "ConnectFour-v0"
BIG_MODEL_NAME = "deepseek/deepseek-r1"

# training parameters
GAMMA = 0.8


def collect_data(model, n_samples: int, use_r1_opponent: float = 0.0):
    full_observations, full_actions, full_rewards = [], [], []
    # now iterate over episodes until we have sufficient data
    while len(full_rewards) < n_samples:
        # run another epifull_sode
        episode_observations, episode_actions = {0: [], 1: []}, {0: [], 1: []} 

        # initialize agents
        model_0 = AnswerTokenAgentWrapper(model)
        model_1 = AnswerTokenAgentWrapper(model if np.random.uniform() > use_r1_opponent else ta.agents.OpenRouterAgent(BIG_MODEL_NAME))
        agents = {0: model_0, 1: model_1}

        # build the enviornment
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

        # get the rewards
        rewards = env.close()

        # check if episode ended in draw
        if rewards[0] == 1 or rewards[1] == 1:
            # calculate the discounted reward and extend to the full tracking listsif len(episode_actions[player_id]) > 0:
            for player_id in range(2):
                T = len(episode_actions[player_id])
                
                # Apply temporal decay weighting from SPAG paper
                # r(t) = (1-γ)γ^(T-t)/(1-γ^(T+1))
                time_weights = np.array([(1-GAMMA) * GAMMA**(T-t-1) for t in range(T)])
                # Normalize weights to sum to 1
                time_weights = time_weights / (1 - GAMMA**(T))
                
                # Apply weights to the final reward
                discounted_player_rewards = rewards[player_id] * time_weights
                
                # Extend to full tracking lists
                full_observations.extend(episode_observations[player_id])
                full_actions.extend(episode_actions[player_id])
                full_rewards.extend(discounted_player_rewards)

    return full_observations, full_actions, full_rewards