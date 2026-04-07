# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Customer Agency Env Environment.

This module creates an HTTP server that exposes the CustomerAgencyEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import json
import os
import random
from typing import List, Dict, Any

from openenv import OpenEnv
from ..models import Action, Observation, State, Reward

class CustomerSupportEnv(OpenEnv):
    def __init__(self):
        self.tasks = []
        self.current_task = None
        self.state: State = None
        self.action_type = Action
        self.observation_type = Observation
        self.setup()

    def setup(self):
        # Load tasks from all JSON files in the data directory
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(data_dir, filename), 'r') as f:
                    # Each file can contain a list of tasks or a single task object
                    content = json.load(f)
                    if isinstance(content, list):
                        self.tasks.extend(content)
                    else:
                        self.tasks.append(content)

    def reset(self) -> Observation:
        self.current_task = random.choice(self.tasks)
        self.state = State(
            task_id=self.current_task['task_id'],
            customer_query=self.current_task['query'],
            policy=self.current_task['policy'],
            context=self.current_task['context'],
            expected_resolution=self.current_task['expected_resolution'],
            grader_logic=self.current_task['grader_logic'],
            history=[],
            steps_taken=0,
            done=False
        )
        return Observation(
            customer_query=self.state.customer_query,
            policy=self.state.policy,
            context=self.state.context,
            history=self.state.history
        )

    def step(self, action: Action) -> (Observation, float, bool, Dict[str, Any]):
        if self.state.done:
            return self.reset(), 0.0, True, {}

        self.state.steps_taken += 1
        self.state.history.append(f"Agent: {action.message}")

        reward = self._grade_response(action)
        self.state.done = True  # End episode after one step

        obs = Observation(
            customer_query=self.state.customer_query,
            policy=self.state.policy,
            context=self.state.context,
            history=self.state.history,
            echoed_message=action.message
        )
        return obs, reward.score, self.state.done, {"feedback": reward.feedback}

    def _grade_response(self, action: Action) -> Reward:
        score = 0.0
        feedback = []

        # Grade resolution
        if action.resolution.lower() == self.state.expected_resolution.lower():
            score += 0.5
            feedback.append("Correct resolution.")
        else:
            feedback.append(f"Incorrect resolution. Expected {self.state.expected_resolution}, got {action.resolution}.")

        # Grade message based on grader logic
        if self.state.grader_logic.lower() in action.message.lower():
            score += 0.5
            feedback.append("Response contains required elements.")
        else:
            feedback.append(f"Response missing required elements: {self.state.grader_logic}")
        
        # Clamp score
        score = min(max(score, 0.0), 1.0)

        return Reward(score=score, feedback=" ".join(feedback))

    def get_state(self) -> State:
        return self.state

# FastAPI app creation
from openenv.core.env_server.http_server import create_app

app = create_app(
    CustomerSupportEnv,
    Action,
    Observation,
    env_name="customer_agency_env",
    max_concurrent_envs=1,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

