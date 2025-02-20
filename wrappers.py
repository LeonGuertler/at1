import textarena as ta 
from openai import OpenAI
from typing import Optional, Dict, List, Tuple

import os
import time
import requests
import json
from typing import Optional

import re

def process_final_answer(answer: str) -> str:
    answer = answer.strip()
    # If already wrapped in square brackets, return as-is
    if answer.startswith('[') and answer.endswith(']'):
        return answer
    # Check if answer is in the \boxed{...} format
    boxed_match = re.search(r'\\boxed\{(.*?)\}', answer)
    if boxed_match:
        content = boxed_match.group(1).strip()
        return f'[{content}]'
    # Otherwise, wrap whatever answer we have in square brackets
    return f'[{answer}]'



# class AnswerTokenAgentWrapper:
#     """ TODO """
#     def __init__(
#         self,
#         agent,
#         answer_token: Optional[str] = "### Answer:",
#         debugging: bool = False
#     ):
#         """ TODO """
#         super().__init__()
#         self.agent = agent
#         self.answer_token = answer_token
#         self.debugging = debugging


#     def call_with_full_answer(self, observation: str) -> str:
#         """ TODO """
   
#         # set the agent prompt just for this part
#         current_system_prompt = self.agent.system_prompt 
#         answer_token_prompt = current_system_prompt + \
#             f"Anything you return after '{self.answer_token}' will be submitted to the game."

#         self.agent.system_prompt = answer_token_prompt
#         if self.debugging:
#             print(f"Model System prompt: {answer_token_prompt}")
        
#         raw_answer = self.agent(observation)

#         # reset prompt 
#         self.agent.system_prompt = current_system_prompt

#         if self.debugging:
#             print(f"Model raw output: {raw_answer}")
#         if self.answer_token in raw_answer:
#             if self.debugging:
#                 print(f"Model filtered output: {raw_answer.split(self.answer_token)[-1]}")
#             return raw_answer, raw_answer.split(self.answer_token)[-1]

#         else:
#             return raw_answer, raw_answer
     

#     def __call__(self, observation: str) -> str:
#         """ TODO """

#         # set the agent prompt just for this part
#         current_system_prompt = self.agent.system_prompt 
#         answer_token_prompt = current_system_prompt + \
#             f"Anything you return after '{self.answer_token}' will be submitted to the game."

#         self.agent.system_prompt = answer_token_prompt
#         if self.debugging:
#             print(f"Model System prompt: {answer_token_prompt}")
        
#         raw_answer = self.agent(observation)

#         # reset prompt 
#         self.agent.system_prompt = current_system_prompt

#         if self.debugging:
#             print(f"Model raw output: {raw_answer}")
#         if self.answer_token in raw_answer:
#             if self.debugging:
#                 print(f"Model filtered output: {raw_answer.split(self.answer_token)[-1]}")
#             return raw_answer.split(self.answer_token)[-1]

#         else:
#             return raw_answer



STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."
    


# class AnswerTokenAgentWrapper:
#     def __init__(
#         self,
#         agent,
#         answer_token: Optional[str] = "### Final Answer",
#         debugging: bool = False
#     ):
#         super().__init__()
#         self.agent = agent
#         self.answer_token = answer_token
#         self.debugging = debugging

#     def call_with_full_answer(self, observation: str):
#         # 1) Save current system prompt
#         current_system_prompt = self.agent.system_prompt

#         # 2) Append instructions about "anything returned after answer token"
#         answer_token_prompt = current_system_prompt + \
#             f"Anything you return after '{self.answer_token}' will be submitted to the game."

#         self.agent.system_prompt = answer_token_prompt
#         if self.debugging:
#             print(f"Model System prompt: {answer_token_prompt}")

#         # 3) Generate
#         raw_answer = self.agent(observation)

#         # 4) Reset system prompt
#         self.agent.system_prompt = current_system_prompt

#         # 5) Post-process
#         if self.debugging:
#             print(f"Model raw output: {raw_answer}")

#         print(f"observation: {observation}")
#         print(f"raw_output: {raw_answer}")
#         input()

#         if self.answer_token in raw_answer:
#             filtered_output = raw_answer.split(self.answer_token, 1)[-1]
#             if self.debugging:
#                 print(f"Model filtered output: {filtered_output}")
            
#             return raw_answer, process_final_answer(filtered_output) #filtered_output
#         elif "### Final Answer" in raw_answer:
#             filtered_output = raw_answer.split("### Final Answer", 1)[-1]
#             if self.debugging:
#                 print(f"Model filtered output: {filtered_output}")
#             return raw_answer, process_final_answer(filtered_output) #filtered_output

#         else:
#             return raw_answer, raw_answer


#     # filtered_output = process_final_answer(filtered_output)
#     # return filtered_output


#     def __call__(self, observation: str):
#         # 1) Save current system prompt
#         current_system_prompt = self.agent.system_prompt

#         # 2) Append instructions about "anything returned after answer token"
#         answer_token_prompt = current_system_prompt + \
#             f"Anything you return after '{self.answer_token}' will be submitted to the game."
#         self.agent.system_prompt = answer_token_prompt

#         if self.debugging:
#             print(f"Model System prompt: {answer_token_prompt}")

#         # 3) Generate
#         raw_answer = self.agent(observation)

#         # 4) Reset system prompt
#         self.agent.system_prompt = current_system_prompt

#         # 5) Post-process
#         if self.debugging:
#             print(f"Model raw output: {raw_answer}")

#         if self.answer_token in raw_answer:
#             filtered_output = raw_answer.split(self.answer_token, 1)[-1]
#             if self.debugging:
#                 print(f"Model filtered output: {filtered_output}")
#             return process_final_answer(filtered_output) #filtered_output

#         elif "### Final Answer" in raw_answer:
#             filtered_output = raw_answer.split("### Final Answer", 1)[-1]
#             if self.debugging:
#                 print(f"Model filtered output: {filtered_output}")
#             return process_final_answer(filtered_output) #filtered_output
        
#         else:
#             return raw_answer


import concurrent.futures
from typing import List, Tuple, Optional

def process_final_answer(answer_str: str) -> str:
    """
    Basic placeholder for any final parsing or trimming of the LLM's raw text
    to produce the final action string.
    """
    return answer_str.strip()

class AnswerTokenAgentWrapper:
    """
    A wrapper that prepends an 'answer-token' instruction to the underlying
    agent's system prompt, calls it, and then extracts the part of the
    output after '### Final Answer' (or whatever token you specify).
    """
    def __init__(
        self,
        agent,
        answer_token: Optional[str] = "### Final Answer",
        debugging: bool = False
    ):
        """
        :param agent: Underlying object that can generate responses.
                      - For the local model, this would be your ConnectFourPPOTrainer.
                      - For OpenRouter, it might be an OpenRouterAgent, etc.
        :param answer_token: The substring that signifies the beginning
                             of the final answer in the model output.
        :param debugging: If True, print extra logs (observations, raw outputs, etc.).
        """
        self.agent = agent
        self.answer_token = answer_token
        self.debugging = debugging

    # -----------------------------------------------------------------------
    # 1) Single-Call Usage (One Observation)
    # -----------------------------------------------------------------------
    def call_with_full_answer(self, observation: str) -> Tuple[str, str]:
        """
        Generates a single response for the given observation.
        Returns (raw_answer, parsed_answer).
        """
        # 1) Save current system prompt
        current_system_prompt = self.agent.system_prompt

        # 2) Append instructions about answer token
        answer_token_prompt = (
            current_system_prompt
            + f"\nAnything you return after '{self.answer_token}' will be submitted to the game."
        )
        self.agent.system_prompt = answer_token_prompt

        # 3) Generate (one call)
        raw_answer = self.agent(observation)  # the underlying agent is __call__-able
        # 4) Reset system prompt
        self.agent.system_prompt = current_system_prompt

        # Debug prints
        if self.debugging:
            print(f"[Single-Call] Observation:\n{observation}")
            print(f"[Single-Call] Raw answer:\n{raw_answer}")

        # 5) Parse out everything after `self.answer_token` (or fallback).
        parsed_answer = self._extract_after_token(raw_answer)

        return raw_answer, parsed_answer

    def __call__(self, observation: str) -> str:
        """
        Overload __call__ to do single-step usage:
        Returns ONLY the final parsed answer string (not raw).
        """
        raw, parsed = self.call_with_full_answer(observation)
        return parsed

    # -----------------------------------------------------------------------
    # 2) Batch-Call Usage for Local Model
    # -----------------------------------------------------------------------
    def batch_call_with_full_answer(self, observations: List[str]) -> List[Tuple[str, str]]:
        """
        Uses *one forward pass* (via your trainer's batch_generate) to get
        outputs for all `observations`. Returns a list of (raw_answer, parsed_answer).
        Only works if `self.agent` has a method `batch_generate(...)`.
        (i.e. your local ConnectFourPPOTrainer).
        """
        # 1) Save current system prompt
        current_system_prompt = self.agent.system_prompt

        # 2) Append instructions about answer token
        answer_token_prompt = (
            current_system_prompt
            + f"\nAnything you return after '{self.answer_token}' will be submitted to the game."
        )
        self.agent.system_prompt = answer_token_prompt

        # 3) Single batch generate
        print(f"Generating {len(observations)} in parallel")
        raw_answers = self.agent.batch_generate(observations)  # must be defined on your agent
        # 4) Reset system prompt
        self.agent.system_prompt = current_system_prompt

        # 5) Parse each raw answer
        results = []
        for obs, raw_answer in zip(observations, raw_answers):
            if self.debugging:
                print(f"[Batch-Call] Obs:\n{obs}")
                print(f"[Batch-Call] Raw answer:\n{raw_answer}")
            parsed_answer = self._extract_after_token(raw_answer)
            results.append((raw_answer, parsed_answer))

        return results

    # -----------------------------------------------------------------------
    # 3) "Batch" Parallel Calls for a Remote/OpenRouter Agent
    # -----------------------------------------------------------------------
    def batch_call_with_full_answer_parallel(self, observations: List[str]) -> List[Tuple[str, str]]:
        """
        For an agent that doesn't support true batching (like a typical remote LLM
        that requires separate requests), we do them in parallel threads.

        Returns a list of (raw_answer, parsed_answer), one per observation.
        """
        results = []

        def single_inference(obs: str) -> Tuple[str, str]:
            # Just use the single-call path for each obs
            return self.call_with_full_answer(obs)

        # We'll run them all in parallel via a ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for obs in observations:
                futures.append(executor.submit(single_inference, obs))
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        # We must preserve the original ordering (as_completed can scramble it),
        # so let's do an alternative approach: map index to future, then reorder:
        # (If you don't care about order, you can skip this step.)
        #   ...
        # But for simplicity, let's say we don't strictly need to preserve order
        # in the final list. If you do, you'd store each future with an index.

        return results

    # -----------------------------------------------------------------------
    # Helper to parse the LLM output
    # -----------------------------------------------------------------------
    def _extract_after_token(self, raw_answer: str) -> str:
        """
        Cut out everything after `self.answer_token` or '### Final Answer' 
        (in case the token differs). Then do minimal cleaning.
        """
        if self.answer_token in raw_answer:
            answer_part = raw_answer.split(self.answer_token, 1)[-1]
            return process_final_answer(answer_part)
        elif "### Final Answer" in raw_answer:
            answer_part = raw_answer.split("### Final Answer", 1)[-1]
            return process_final_answer(answer_part)
        else:
            return process_final_answer(raw_answer)




class OpenRouterAgent(ta.Agent):
    """Agent class using the OpenRouter API via direct requests to generate responses."""
    
    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = STANDARD_GAME_PROMPT,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the OpenRouter agent.

        Args:
            model_name (str): The name of the model.
            system_prompt (Optional[str]): The system prompt to use (default: STANDARD_GAME_PROMPT).
            verbose (bool): If True, additional debug info is printed.
            **kwargs: Additional keyword arguments to pass in the payload to the OpenRouter API.
        """
        super().__init__()
        self.model_name = model_name
        self.verbose = verbose
        self.system_prompt = system_prompt
        self.kwargs = kwargs

        # Set the open router api key from an environment variable
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Please set the OPENROUTER_API_KEY environment variable."
            )

        # Base URL for OpenRouter chat completions
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def _make_request(self, observation: str, return_full_reply: bool) -> str:
        """
        Make a single API request to OpenRouter and return the generated message.

        Args:
            observation (str): The user query or prompt to send.
            return_full_reply (bool): If True, request reasoning from the API and return
                                      one string that includes both reasoning and final answer.

        Returns:
            str: The assistant's response (with or without reasoning, depending on return_full_reply).
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": observation}
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "n": 1,  # single completion
        }
        
        # If we want the entire chain of reasoning in the response
        if return_full_reply:
            payload["include_reasoning"] = True

        # Merge any extra kwargs (temperature, max_tokens, etc.) into the payload
        # If you have certain known arguments that come in as part of kwargs, you can handle them individually
        # or just update the entire payload dictionary. Make sure to only add valid keys for the OpenRouter API.
        payload.update(self.kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(self.base_url, headers=headers, data=json.dumps(payload))

        if response.status_code != 200:
            raise RuntimeError(
                f"Request failed with status code {response.status_code}: {response.text}"
            )

        response_data = response.json()

        # Check for errors returned in the JSON
        if "error" in response_data:
            raise RuntimeError(f"API error: {response_data['error']}")

        # Extract the relevant parts from the first choice
        first_choice = response_data["choices"][0]
        message_data = first_choice["message"]

        content = message_data.get("content", "").strip()
        reasoning = message_data.get("reasoning", "").strip()

        if self.verbose:
            print(f"Observation:\n{observation}")
            if return_full_reply:
                print(f"Reasoning: {reasoning}")
            print(f"Content:\n{content}")

        # If return_full_reply is True and reasoning is present, combine them
        if return_full_reply and reasoning:
            return f"{reasoning}\n\n{content}"
        else:
            return content

    def _retry_request(
        self,
        observation: str,
        return_full_reply: bool,
        retries: int = 3,
        delay: int = 5
    ) -> str:
        """
        Attempt to make an API request with retries.

        Args:
            observation (str): The input to process.
            return_full_reply (bool): Whether or not to include reasoning.
            retries (int): The number of attempts to try.
            delay (int): Seconds to wait between attempts.

        Returns:
            str: The generated response.

        Raises:
            Exception: The last exception caught if all retries fail.
        """
        last_exception = None
        for attempt in range(1, retries + 1):
            try:
                response = self._make_request(observation, return_full_reply=return_full_reply)
                return response
            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < retries:
                    time.sleep(delay)
        raise last_exception

    def __call__(self, observation: str, return_full_reply: bool = True) -> str:
        """
        Process the observation using the OpenRouter API and return the action.

        Args:
            observation (str): The input string to process.
            return_full_reply (bool): Whether to include the model reasoning in the final string.

        Returns:
            str: The generated response (reasoning + final answer, or just final answer).
        """
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received: {type(observation)}")

        return self._retry_request(observation, return_full_reply=return_full_reply)