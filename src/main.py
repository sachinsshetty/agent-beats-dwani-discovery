import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
import dwani 

# Configure Dwani/OpenAI-compatible credentials
dwani.api_key = os.getenv("DWANI_API_KEY")
dwani.api_base = os.getenv("DWANI_API_BASE_URL") # Ensure this ends in /v1

# Fix: Prepend 'openai/' to the model name
# This tells LiteLLM to use the OpenAI provider logic for the Dwani base URL
openai_model = LiteLlm(
    model="openai/gemma3", 
    api_base=dwani.api_base,
    api_key=dwani.api_key
)

agent = Agent(
    name="OpenAIAssistant",
    model=openai_model,
    instruction="You are a helpful assistant powered by OpenAI running inside Google's ADK.",
)

async def main():
    app_name = "dwani_discovery_app"
    user_id = "sachin_dev"
    session_id = "current_session_001"
    
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name=app_name
    )

    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    user_query = "What are the benefits of using a model-agnostic framework?"
    new_message = Content(role="user", parts=[Part(text=user_query)])

    print("Agent Response: ", end="", flush=True)
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print(event.content.parts[0].text)

if __name__ == "__main__":
    asyncio.run(main())
