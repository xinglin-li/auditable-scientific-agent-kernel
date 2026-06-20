# src/agent_runtime/graph/service.py
from typing import Any, Dict, List
from langgraph.types import Command
from agent_runtime.graph.persistence import CheckpointPersistenceManager
from agent_runtime.graph.builder import create_macro_agent_graph

class MacroAgentGraphService:
    def __init__(self, db_path: str = ":memory:"):
        self._persistence = CheckpointPersistenceManager.create_sqlite_checkpointer(db_path)
        self.checkpointer = self._persistence.__enter__()
        self.graph = create_macro_agent_graph(checkpointer=self.checkpointer)

    def close(self) -> None:
        self._persistence.__exit__(None, None, None)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def run_workflow(self, thread_id: str, run_id: str, user_query: str) -> Dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {"thread_id": thread_id, "run_id": run_id, "user_query": user_query, "status": "running"}
        
        # Invoke the graph normally. An interrupt pauses execution here and persists the
        # latest checkpoint; invoke then returns the paused state.
        return self.graph.invoke(initial_state, config=config)

    def resume_workflow(self, thread_id: str, review_action: dict) -> Dict[str, Any]:
        """Resume an interrupted thread with an external human-review decision."""
        config = {"configurable": {"thread_id": thread_id}}
        
        # Use langgraph.types.Command to pass resume data to the interrupted node.
        resume_command = Command(resume=review_action)
        
        # Invoke with the command and thread-scoped configuration to continue execution.
        return self.graph.invoke(resume_command, config=config)

    def get_state_history(self, thread_id: str) -> List[Dict[str, Any]]:
        config = {"configurable": {"thread_id": thread_id}}
        history_chain = []
        for state_snapshot in self.graph.get_state_history(config):
            history_chain.append({
                "checkpoint_id": state_snapshot.config["configurable"].get("checkpoint_id"),
                "values": state_snapshot.values,
                "next_nodes": state_snapshot.next,
            })
        return history_chain
