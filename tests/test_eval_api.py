# tests/test_eval_api.py
import pytest
import json
from fastapi.testclient import TestClient
from agent_runtime.api import app

def test_sse_evaluation_streaming_endpoint():
    """端到端微服务集成测试：验证批量评测接口的 SSE 握手、并发协同与流式解析完整性"""
    
    client = TestClient(app)
    
    # 1. 构造一个包含 2 个典型评测 Task 的微型数据集契约
    payload = {
        "tasks": [
            {
                "task_id": "api_test_task_01",
                "name": "时序统计分析",
                "user_input": "Calculate RMSE",
                "expected_outcome": {},
                "trajectory_rules": [],
                "limits": {"max_steps": 3}
            },
            {
                "task_id": "api_test_task_02",
                "name": "回测参数推演",
                "user_input": "Optimize params",
                "expected_outcome": {},
                "trajectory_rules": [],
                "limits": {"max_steps": 2}
            }
        ],
        "num_trials": 2,          # 总共应当触发 2 * 2 = 4 次独立异步 Trials
        "initial_concurrency": 2
    }
    
    # 2. 交付 HTTP POST 评测网关，使用 Streaming 方式拦截响应
    with client.stream("POST", "/api/v1/evals/run", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        
        received_trials = []
        has_done_signal = False
        
        # 3. 逐行对网络传输的 SSE 字节流进行解包与审计
        for line in response.iter_lines():
            if not line:
                continue
                
            decoded_line = line if isinstance(line, str) else line.decode("utf-8")
            if decoded_line.startswith("data: "):
                data_content = decoded_line[len("data: "):]
                
                if data_content == "[DONE]":
                    has_done_signal = True
                    break
                    
                # 能够成功进行 JSON 逆序列化，证明微服务输出格式完美符合 SSE 规约
                frame = json.loads(data_content)
                assert "task_id" in frame
                assert "trial_id" in frame
                assert "all_passed" in frame
                
                received_trials.append(frame)
                
        # 4. 终极断言验证：4 次并发设计的独立 Trial 必须一个不少地全部流式输出完毕
        assert len(received_trials) == 4
        assert has_done_signal is True
