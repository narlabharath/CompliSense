from langgraph.graph import StateGraph
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import Image, display
import json

# ✅ Step 1: Define the state
class TranscriptState(dict):
    pass

# ✅ Step 2: Transcript input
transcript = """
[00:01] Agent: Let's discuss FX Forward and Market Risk.
[00:12] Customer: Okay, what are the risks involved?
[00:25] Agent: There's liquidity risk for this product as well.
[00:40] Agent: Now moving on to Bonds (Secondary)...
"""

# ✅ Step 3: Static Entity Extractor Node
def extract_entities_node(state: TranscriptState) -> TranscriptState:
    print("📎 Extracting entities (static mode)...")
    state["products_discussed"] = [
        {
            "name": "FX Forward",
            "start_time": "00:01",
            "end_time": "00:20",
            "mentions": ["00:01", "00:10", "00:18"]
        },
        {
            "name": "Bonds (Secondary)",
            "start_time": "00:40",
            "end_time": "00:50",
            "mentions": ["00:40"]
        }
    ]
    state["disclosures_mentioned"] = ["Market Risk", "Liquidity Risk"]
    print(f"State after extract_entities: {state}")
    return state

# ✅ Step 4: Final Node
def complete_node(state: TranscriptState) -> TranscriptState:
    print("✅ Flow complete.")
    state["status"] = "done"
    print(f"State after complete_node: {state}")
    return state

# ✅ Step 5: Build the LangGraph
builder = StateGraph(TranscriptState)
builder.add_node("extract_entities", extract_entities_node)
builder.add_node("complete", complete_node)
builder.set_entry_point("extract_entities")
builder.add_edge("extract_entities", "complete")
builder.set_finish_point("complete")
graph = builder.compile()

# ✅ Step 6: Run the Graph
initial_state = TranscriptState({"transcript": transcript})
final_state = graph.invoke(initial_state)

# ✅ Step 7: Display Final State
if final_state is None:
    final_state = initial_state

print("\n📦 Final Output:")
print(json.dumps(final_state, indent=2))

# ✅ Step 8: Generate and Show Mermaid Graph
image_bytes = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
with open("langgraph_flow.png", "wb") as f:
    f.write(image_bytes)

print("✅ Mermaid graph saved as 'langgraph_flow.png'")
display(Image("langgraph_flow.png"))
