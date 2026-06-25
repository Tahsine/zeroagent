import asyncio
from zeroagent import Agent, tool, LLMClient

@tool(description="Additionne deux nombres entiers")
def add(a:int, b:int) -> int:
    return a + b

llm = LLMClient(
    base_url="https://ollama.com/v1",
    model="minimax-m3:cloud",
    api_key="4d521aaed302428cb3a4de03eda0fbc9.4jOowXaZajJocowLY-Qa-5ME"
)

agent = Agent(llm=llm, tools=[add], verbose=True)

async def main():
    # Test 1 — réponse directe
    print("=== Test 1 : réponse directe ===")
    result = await agent.arun("Quelle est la capitale du Bénin ?")
    print("→", result)

    # Test 2 — avec tool call
    print("\n=== Test 2 : tool call ===")
    result = await agent.arun("Calcule 42 + 58")
    print("→", result)

    # Test 3 — run_full pour voir les métadonnées
    print("\n=== Test 3 : arun_full ===")
    result = await agent.arun_full("Calcule 100 + 200")
    print("→ answer:", result.answer)
    print("→ stop_reason:", result.stop_reason)
    print("→ iterations:", result.iterations)
    print("→ tool_calls_made:", result.tool_calls_made)

asyncio.run(main())