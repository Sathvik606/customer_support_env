# Customer Support Agent Environment

This project is an OpenEnv-compliant reinforcement learning environment that simulates a real-world customer support scenario. An AI agent learns to handle customer queries by following company policies to resolve issues related to refunds, replacements, deliveries, and payments.

## Environment Description and Motivation

The goal of this environment is to train and evaluate AI agents on their ability to perform policy-based issue resolution in a customer support context. This is a critical task for many businesses, and automating it with AI can lead to significant efficiency gains and improved customer satisfaction. This environment provides a realistic and challenging testbed for developing such agents.

## Action and Observation Spaces

### Action Space

The agent's action is defined by the `Action` model in `models.py`:

-   `resolution` (str): The resolution action to take (e.g., 'approve', 'deny', 'escalate').
-   `message` (str): The message to send to the customer.

### Observation Space

The agent receives an observation defined by the `Observation` model in `models.py`:

-   `customer_query` (str): The customer's query.
-   `policy` (Dict[str, Any]): The applicable company policy.
-   `context` (Dict[str, Any]): Additional context for the query.
-   `history` (List[str]): The history of interactions in the current episode.
-   `echoed_message` (Optional[str]): The last message sent by the agent.

## Task Descriptions

The environment includes tasks of varying difficulty across four categories:

### 1. Refund Resolution
-   **Easy**: A straightforward refund request within the policy limits.
-   **Medium**: A request that is outside the standard refund window but may qualify for a different resolution (e.g., replacement).
-   **Hard**: A request for a non-refundable item, requiring the agent to deny the refund while potentially offering an alternative solution.

### 2. Replacement Handling
-   Tasks will involve deciding whether a product qualifies for replacement based on policy rules.

### 3. Delivery Information Support
-   Tasks will involve handling delivery-related queries, including delays and tracking issues.

### 4. Payment Issue Resolution
-   Tasks will involve addressing payment-related concerns such as duplicate charges or failed transactions.

## Setup and Usage

### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sathvik606/customer_support_env.git
    cd customer_support_env
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r server/requirements.txt
    ```

3.  **Run the environment server:**
    ```bash
    uvicorn server.app:app --host 0.0.0.0 --port 8000
    ```

### Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t customer-agency-env .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 customer-agency-env
    ```

## How to Run Inference

The `inference.py` script runs a baseline agent against the environment.

1.  **Set environment variables:**
    ```bash
    export API_BASE_URL="your_llm_api_endpoint"
    export MODEL_NAME="your_model_name"
    export HF_TOKEN="your_api_key"
    ```

2.  **Run the script:**
    ```bash
    python inference.py
    ```

## Baseline Scores

The baseline scores will be populated here after running the inference script against a set of evaluation tasks.

| Task ID         | Category | Difficulty | Score |
|-----------------|----------|------------|-------|
| refund_easy_1   | refund   | easy       | TBD   |
| refund_medium_1 | refund   | medium     | TBD   |
| refund_hard_1   | refund   | hard       | TBD   |

## Deployment to Hugging Face Spaces

This environment is designed to be deployed as a Hugging Face Space.

1.  Create a new Space on Hugging Face, selecting the "Docker" template.
2.  Link the Space to your GitHub repository.
3.  The `Dockerfile` and `openenv.yaml` are configured for deployment. Hugging Face will automatically build and deploy the environment.

openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**CustomerAgencyAction**: Contains a single field
- `message` (str) - The message to echo back

### Observation
**CustomerAgencyObservation**: Contains the echo response and metadata
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a Customer Agency Env environment server running, you can connect directly:

```python
from customer_agency_env import CustomerAgencyEnv

# Connect to existing server
customer_agency_envenv = CustomerAgencyEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = customer_agency_envenv.reset()
result = customer_agency_envenv.step(CustomerAgencyAction(message="Hello!"))
```

Note: When connecting to an existing server, `customer_agency_envenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from customer_agency_env import CustomerAgencyAction, CustomerAgencyEnv

# Connect with context manager (auto-connects and closes)
with CustomerAgencyEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(CustomerAgencyAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    CustomerAgencyEnvironment,  # Pass class, not instance
    CustomerAgencyAction,
    CustomerAgencyObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from customer_agency_env import CustomerAgencyAction, CustomerAgencyEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with CustomerAgencyEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(CustomerAgencyAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/customer_agency_env_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
customer_agency_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # CustomerAgencyEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── customer_agency_env_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
