# tests/test_eval_api.py
import pytest
import json
from fastapi.testclient import TestClient
from agent_runtime.api import app

def test_sse_evaluation_streaming_endpoint():
    """Verify SSE negotiation, concurrency, and parsing for the evaluation API."""
    
    client = TestClient(app)
    
    # 1. Build a small dataset containing two representative evaluation tasks.
    payload = {
        "tasks": [
            {
                "task_id": "api_test_task_01",
                "name": "Time-series statistical analysis",
                "user_input": "Calculate RMSE",
                "expected_outcome": {},
                "trajectory_rules": [],
                "limits": {"max_steps": 3}
            },
            {
                "task_id": "api_test_task_02",
                "name": "Backtest parameter analysis",
                "user_input": "Optimize params",
                "expected_outcome": {},
                "trajectory_rules": [],
                "limits": {"max_steps": 2}
            }
        ],
        "num_trials": 2,          # Two tasks by two repetitions produce four trials.
        "initial_concurrency": 2
    }
    
    # 2. Submit the POST request and consume the response as a stream.
    with client.stream("POST", "/api/v1/evals/run", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        
        received_trials = []
        has_done_signal = False
        
        # 3. Decode and inspect the SSE stream line by line.
        for line in response.iter_lines():
            if not line:
                continue
                
            decoded_line = line if isinstance(line, str) else line.decode("utf-8")
            if decoded_line.startswith("data: "):
                data_content = decoded_line[len("data: "):]
                
                if data_content == "[DONE]":
                    has_done_signal = True
                    break
                    
                # Successful JSON decoding confirms a valid SSE payload.
                frame = json.loads(data_content)
                assert "task_id" in frame
                assert "trial_id" in frame
                assert "all_passed" in frame
                
                received_trials.append(frame)
                
        # 4. Confirm that all four independent trials were streamed.
        assert len(received_trials) == 4
        assert has_done_signal is True
