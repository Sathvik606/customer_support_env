from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class Action(BaseModel):
    """
    The action that the agent takes in response to an observation.
    """
    resolution: str = Field(description="The resolution action to take, e.g., 'approve', 'deny', 'escalate'.")
    message: str = Field(description="The message to send to the customer.")

class Observation(BaseModel):
    """
    The observation that the agent receives from the environment.
    """
    customer_query: str = Field(description="The customer's query.")
    policy: Dict[str, Any] = Field(description="The applicable company policy.")
    context: Dict[str, Any] = Field(description="Additional context for the query.")
    history: List[str] = Field(description="The history of interactions in the current episode.")
    echoed_message: Optional[str] = Field(None, description="The last message sent by the agent.")

class State(BaseModel):
    """
    The complete state of the environment.
    """
    task_id: str
    customer_query: str
    policy: Dict[str, Any]
    context: Dict[str, Any]
    expected_resolution: str
    grader_logic: str
    history: List[str]
    steps_taken: int
    done: bool

class Reward(BaseModel):
    """
    The reward signal for the agent.
    """
    score: float = Field(ge=0.0, le=1.0)
    feedback: str

