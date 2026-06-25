import asyncio
from zeroagent import Agent, tool, LLMClient
from zeroagent.workflow import Graph, Node, State

# --- Tools ---
@tool(description="Additionne deux nombres")
def add(a: int, b: int) -> int:
    return a + b

# --- Agent dans un nœud ---
llm = LLMClient(
    base_url="https://ollama.com/v1",
    model="minimax-m3:cloud",
    api_key="4d521aaed302428cb3a4de03eda0fbc9.4jOowXaZajJocowLY-Qa-5ME",
)
agent = Agent(llm=llm, tools=[add])

# --- Nœuds purs ---
def prepare(state: State) -> State:
    state["input"] = f"Calcule {state['a']} + {state['b']}"
    return state

def format_output(state: State) -> State:
    state["final"] = f"Résultat : {state['output']}"
    return state

# --- Graph ---
graph = Graph(name="demo")
graph.add_node(Node("prepare", fn=prepare))
graph.add_node(Node("agent", fn=agent))
graph.add_node(Node("format", fn=format_output))
graph.add_edge("prepare", "agent")
graph.add_edge("agent", "format")
graph.set_entry("prepare")
graph.set_finish("format")

print(graph.visualize())

result = graph.run(State({"a": 42, "b": 58}))
print(result.state["final"])
print("Nodes exécutés:", result.executed)