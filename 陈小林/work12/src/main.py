from agent import Agent
from src.models.entities import LLMConfig
def main():
    config = LLMConfig(model_type="deepseek",
                       model="deepseek-chat",
                       api_key="sk-16048b65b2ab45dc9f15d011e0cabacd")
    agent = Agent(llm_config=config)
    agent.start(session_id='1')

if __name__ == '__main__':
    main()