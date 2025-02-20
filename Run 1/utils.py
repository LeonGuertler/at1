import textarena as ta 
import os, re, time, requests, json
from typing import Tuple, List

from config import STANDARD_GAME_PROMPT



def process_final_answer(answer: str) -> str:
    """ By default trained on math, so the output format is slightly off """
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


# Answer token agent wrapper with full reply return
class AnswerTokenAgentWrapper:
    def __init__(self, agent, answer_token="### Final Answer"):
        self.agent = agent
        self.answer_token = answer_token 
        self.answer_token_prompt = f"\nAnything you return after '{self.answer_token}' will be submitted to the game."

    def _extract_after_token(self, raw_answer: str) -> str:
        if raw_answer is None:
            return "No Answer"
        if self.answer_token in raw_answer:
            answer_part = raw_answer.split(self.answer_token, 1)[-1]
            return process_final_answer(answer_part)
        else:
            return process_final_answer(raw_answer)

    def call_with_full_answer(self, observation: str) -> Tuple[str, str]:
        current_system_prompt = self.agent.system_prompt
        self.agent.system_prompt = current_system_prompt + self.answer_token_prompt 

        # generate raw answer
        raw_answer = self.agent(observation)
        # input(raw_answer)
        self.agent.system_prompt = current_system_prompt
        return raw_answer, self._extract_after_token(raw_answer) 

    def __call__(self, observation: str) -> str:
        raw, parsed = self.call_with_full_answer(observation)
        return parsed 

    def batch_call_with_full_answer(self, observations: List[str]) -> List[Tuple[str, str]]:
        current_system_prompt = self.agent.system_prompt
        self.agent.system_prompt = current_system_prompt + self.answer_token_prompt 

        # generate raw answers
        raw_answers = self.agent.batch_generate(observations)
        self.agent.system_prompt = current_system_prompt
        return [(raw_answer, self._extract_after_token(raw_answer)) for raw_answer in raw_answers]

    # def batch_call_with_full_answer_parallel(self, observations: List[str]) -> List[Tuple[str, str]]:
    #     """ to call the openrouter models in parallel """
    #     results = []

    #     def single_inference(obs: str) -> Tuple[str, str]:
    #         # Just use the single-call path for each obs
    #         return self.call_with_full_answer(obs)

    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         futures = []
    #         for obs in observations:
    #             futures.append(executor.submit(single_inference, obs))
            
    #         for future in concurrent.futures.as_completed(futures):
    #             results.append(future.result())
        
    #     return results 




# modified openrouter class to allow for returned reasoning chain

class OpenRouterAgent(ta.Agent):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.system_prompt = STANDARD_GAME_PROMPT

        # Set the open router api key from an environment variable
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable.")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def _make_request(self, observation: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": observation}
        ]

        payload = {"model": self.model_name, "messages": messages, "n": 1, "include_reasoning": True}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        # print('call model')
        response = requests.post(self.base_url, headers=headers, data=json.dumps(payload))
        # print('answer received')
        if response.status_code != 200:
            return "No action"
            raise RuntimeError(f"Request failed with status code {response.status_code}: {response.text}")
        response_data = response.json()
        if "error" in response_data:
            return "No action"
            raise RuntimeError(f"API error: {response_data['error']}")

        # Extract the relevant parts from the first choice
        first_choice = response_data["choices"][0]
        message_data = first_choice["message"]

        content = message_data.get("content", "").strip()
        reasoning = message_data.get("reasoning", "").strip()

        return f"{reasoning}\n\n{content}"

    def _retry_request(
        self,
        observation: str,
        retries: int = 3,
        delay: int = 2
    ) -> str:
        last_exception = None
        for attempt in range(1, retries + 1):
            try:
                response = self._make_request(observation)
                return response
            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < retries:
                    time.sleep(delay)
        raise last_exception

    def __call__(self, observation: str) -> str:
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received: {type(observation)}")
        response = self._retry_request(observation)
        if response is None:
            return "No reply provided"
        else: 
            return response