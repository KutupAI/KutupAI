# Worker Agents Layer

Hard rule: no Agent connects to Storage directly. Every Agent returns its result to graph_state; state_manager.py in Orchestration is what persists it.
