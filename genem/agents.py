from openai import OpenAI
import openai
import time
from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np
from openai import OpenAI
import openai
import time
from primitives.primitive import Primitive
import re

class Chatbot:
    def __init__(self, system_prompt=None, model="gpt-4.1", api_key=None, max_retries=3):
        """
        Chat-based LLM with persistent conversation history.
        """
        if api_key:
            openai.api_key = api_key

        self.client = OpenAI()
        self.model = model
        self.max_retries = max_retries

        # Initialize history
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def query(self, prompt, temperature=0.7, max_tokens=10_000):
        """
        Sends a prompt to the model while maintaining chat history.
        """
        # Add user message to history
        self.messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                reply = response.choices[0].message.content

                # Save assistant reply in history
                self.messages.append({"role": "assistant", "content": reply})

                return reply

            except openai.RateLimitError:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
            except Exception as e:
                raise RuntimeError(f"OpenAI request failed: {e}")

    def reset(self):
        """Clears conversation history (except system prompt)."""
        sys_msg = next((m for m in self.messages if m["role"] == "system"), None)
        self.messages = [sys_msg] if sys_msg else []

    def get_history(self):
        """Returns a copy of the conversation history."""
        return list(self.messages)
    

class SocialExpression(Chatbot):
    
    def __init__(self, api_key, prompt=Path('genem/prompts/social_understanding.txt')):
        with open(prompt, 'r') as f:
            system_prompt = f.read()
        super().__init__(
            system_prompt = system_prompt,
            api_key       = api_key
        )
    
    def query(self, human_scenario):
        response = super().query(human_scenario)
        parts = response.split('===')
        if len(parts) != 2:
            raise Exception(f'Illegal prompt response {response}')
        return parts[-1]
    
class TrajectoryGenerator(Chatbot):
    
    def __init__(
        self, 
        api_key, 
        primitives: list[Primitive],
        prompt=Path('genem/prompts/robot_expressive.txt')
    ):
        with open(prompt, 'r') as f:
            system_prompt = f.read()
        self.primitives = primitives
        primitives_descriptions = [
            p.get_name() + ': ' + p.get_description() for p in primitives
        ]
        primitives_descriptions = ['PRIMITIVES LIST'] + primitives_descriptions
        final_system_prompt = system_prompt + '\n' + '\n'.join(primitives_descriptions)
        super().__init__(
            system_prompt = final_system_prompt,
            api_key       = api_key
        )
    def parse_actions(self, s):
        pattern = r"-\s*([A-Za-z_]+)\[([0-9]*\.?[0-9]+)\]"
        actions = []
        durations = []
        
        for action, duration in re.findall(pattern, s):
            actions.append(action)
            durations.append(float(duration))
        
        return actions, durations
    
    def get_primitive_list(self, primitives_list, durations_list):
        ret = []
        for p, d in zip(primitives_list, durations_list):
            for p_obj in self.primitives:
                if p == p_obj.get_name():
                    ret.append(p_obj(d))
                    break
        
        return ret

    def query(self, expressive_description):
        response = super().query(expressive_description)
        parts = response.split('===')
        if len(parts) != 2:
            raise Exception(f'Illegal prompt response {response}')
        
        print(parts[-1])
        actions, durations = self.parse_actions(parts[-1])
        print(actions, durations)
        p_list = self.get_primitive_list(actions, durations)
        return p_list


if __name__ == '__main__':
    load_dotenv()
    test = 'gen-em'
    if test == 'chatbot':
        agent = Chatbot(
            system_prompt='when you speak, put a hyphen "-" between each word. ie. hyphens replace spaces, but words themselves remain the same',
            api_key=os.getenv("OPENAI_API_KEY")
        )
        response = agent.query(prompt='hello world')
        print(response)
        response = agent.query(prompt='good. what did i just say to you?')
        print(response)
    elif test == 'gen-em':
        from primitives.primitive import PRIMITIVES
        se = SocialExpression(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        se_response = se.query('a human walks into the room for 1 second, waves for 1 second, then leaves in 1 second')
        print(se_response)
        print('Now generating trajectory...')
        tg = TrajectoryGenerator(
            api_key=os.getenv("OPENAI_API_KEY"),
            primitives=PRIMITIVES
        )
        response = tg.query(se_response)
        print(response)