import asyncio
import os
import json
from datetime import datetime
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional

from customer_agency_env.client import CustomerAgencyEnv
from customer_agency_env.models import Action

# --- Constants ---
MAX_STEPS = 10
MAX_TOTAL_REWARD = 1.0 
SUCCESS_SCORE_THRESHOLD = 0.8

# --- Structured Logging ---
def log_start(task_id: str, task_category: str, task_difficulty: str):
    print(json.dumps({
        "event": "start",
        "timestamp": datetime.utcnow().isoformat(),
        "task_id": task_id,
        "task_category": task_category,
        "task_difficulty": task_difficulty,
    }), flush=True)

def log_step(step: int, action: Dict[str, Any], reward: float, done: bool, error: Optional[str]):
    print(json.dumps({
        "event": "step",
        "timestamp": datetime.utcnow().isoformat(),
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "error": error,
    }), flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(json.dumps({
        "event": "end",
        "timestamp": datetime.utcnow().isoformat(),
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
    }), flush=True)

async def get_model_message(client: AsyncOpenAI, model_name: str, step: int, obs: Dict[str, Any], history: List[str]) -> Dict[str, Any]:
    """Generates the agent's response using an LLM."""
    system_prompt = (
        "You are a customer support agent. Your goal is to resolve customer queries "
        "by strictly following the provided company policy. Analyze the customer query, "
        "the policy, and the context, then choose a resolution and write a polite, "
        "professional response to the customer."
        "\nYour output must be a JSON object with two keys: 'resolution' and 'message'."
        "\nThe 'resolution' must be one of the expected actions based on the policy (e.g., 'approve', 'deny', 'escalate')."
        "\nThe 'message' is the response you will send to the customer."
    )

    user_prompt = (
        f"## Policy:\n{json.dumps(obs['policy'], indent=2)}\n\n"
        f"## Context:\n{json.dumps(obs['context'], indent=2)}\n\n"
        f"## Customer Query:\n'{obs['customer_query']}'\n\n"
        f"## History:\n" + "\n".join(history) + "\n\n"
        "Based on the information above, provide the next resolution and message."
    )

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        action_json = json.loads(response.choices[0].message.content)
        return action_json
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", flush=True)
        return {"resolution": "error", "message": "I encountered an internal error."}


async def main():
    """Main function to run the inference script."""
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("HF_TOKEN") # Using HF_TOKEN as the API key as per instructions

    if not all([api_base_url, model_name, api_key]):
        raise ValueError("API_BASE_URL, MODEL_NAME, and HF_TOKEN environment variables must be set.")

    llm_client = AsyncOpenAI(api_key=api_key, base_url=api_base_url)
    
    env = CustomerAgencyEnv(base_url="http://localhost:8000")
    
    steps_taken = 0
    rewards = []
    score = 0.0
    success = False

    try:
        result = await env.reset()
        obs = result.observation
        
        # Fetch task details from the initial state for logging
        env_state = await env.state()
        log_start(
            task_id=env_state.task_id,
            task_category=env_state.policy.get('name', 'Unknown'),
            task_difficulty=env_state.context.get('difficulty', 'Unknown')
        )

        history = [f"System: New case. Customer query: '{obs.customer_query}'"]

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_dict = await get_model_message(llm_client, model_name, step, obs.model_dump(), history)
            
            action = Action(**action_dict)

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action.model_dump(), reward=reward, done=done, error=error)

            history.append(f"Step {step}: Agent chose resolution '{action.resolution}' with message: '{action.message}' -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards)
        score = min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
