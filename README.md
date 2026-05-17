# CLIFF-WALKING-GYMNASIUM-PROJECT-SARSA-IMPLEMENTATION-
1. Table of Contents

Overview
What is SARSA?
Environment
Algorithm Deep Dive
Results
Project Structure
Getting Started
Hyperparameters
Key Takeaways
Further Reading


2. Overview
This project implements SARSA, one of the core Temporal Difference reinforcement learning algorithms, entirely from scratch using only NumPy and OpenAI's Gymnasium. The agent learns to navigate a treacherous grid-world — avoiding a cliff while reaching a goal — purely through trial-and-error interaction with the environment.
No neural networks. No libraries. Just math, a Q-table, and the SARSA update rule.
This is a perfect reference for:

Understanding TD learning vs. Monte Carlo methods
Contrasting on-policy (SARSA) vs. off-policy (Q-Learning) approaches
Building intuition for the explore-exploit tradeoff


🔍 What is SARSA?
SARSA is an on-policy TD control algorithm. It learns the action-value function Q(s, a) by updating estimates using the tuple:
(St, At, Rt+1, St+1, At+1)→SARSA\text{(S}_{t}\text{, A}_{t}\text{, R}_{t+1}\text{, S}_{t+1}\text{, A}_{t+1}\text{)} \quad \rightarrow \quad \text{SARSA}(St​, At​, Rt+1​, St+1​, At+1​)→SARSA
Update Rule
Q(St,At)←Q(St,At)+α[Rt+1+γ⋅Q(St+1,At+1)−Q(St,At)]Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big[ R_{t+1} + \gamma \cdot Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \Big]Q(St​,At​)←Q(St​,At​)+α[Rt+1​+γ⋅Q(St+1​,At+1​)−Q(St​,At​)]
SymbolMeaningα\alpha
αLearning rate — how fast we update estimatesγ\gamma
γDiscount factor — how much we value future rewardsRt+1R_{t+1}
Rt+1​Reward received after taking action AtA_t
At​Q(St+1,At+1)Q(S_{t+1}, A_{t+1})
Q(St+1​,At+1​)Value of the next actual action (on-policy)
SARSA vs. Q-Learning — The Key Distinction
FeatureSARSA (On-Policy)Q-Learning (Off-Policy)Bootstrap targetNext action from current policyNext action from greedy policyRisk behaviorCautious — avoids risky pathsAggressive — optimal but riskyConvergenceTo optimal ε-soft policyTo true optimal policyCliffWalking behaviorSafer path away from cliffOptimal but cliff-adjacent path

On CliffWalking, SARSA famously learns the safe path while Q-Learning discovers the riskier but shorter cliff-edge route — a beautiful illustration of the on-policy vs. off-policy distinction.


🗺️ Environment: CliffWalking-v1
o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o  o  o  o  o
S  C  C  C  C  C  C  C  C  C  C  G
PropertyValueGrid size4 × 12 (48 states)StartBottom-left (state 36)GoalBottom-right (state 47)CliffBottom row, states 37–46ActionsUp (0), Right (1), Down (2), Left (3)Step reward−1Cliff penalty−100 (resets to start)TerminalReaching the goal
The agent must find a path from S to G, learning to avoid the cliff through accumulated experience.

⚙️ Algorithm Deep Dive
Training Loop (Pseudocode)
Initialize Q(s, a) = 0 for all s, a
For each episode:
    Reset environment → get initial state S
    Choose A from S using ε-greedy policy
    
    While episode not done:
        Take action A → observe R, S'
        Choose A' from S' using ε-greedy policy   ← KEY: on-policy step
        
        Q(S,A) ← Q(S,A) + α[R + γ·Q(S',A') - Q(S,A)]
        
        S ← S'
        A ← A'   ← carry the chosen action forward
ε-Greedy Exploration Policy
pythondef epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(Q[state])        # Exploit
The ε-greedy policy balances exploration (trying new actions to discover better paths) against exploitation (using the best-known action). With ε = 0.1, the agent explores 10% of the time throughout training.

3.Results
After 500 training episodes, the agent converges to a safe, reliable policy:

 Consistently reaches the goal without falling off the cliff
 Learns the longer but safer route (characteristic of SARSA's on-policy nature)
 Q-table values reflect accurate estimates of state-action quality

Training progression (typical output):
episode=1/500:   total reward = -748  & ep length = 252
episode=50/500:  total reward = -127  & ep length = 31
episode=100/500: total reward = -21   & ep length = 21
episode=250/500: total reward = -17   & ep length = 17
episode=500/500: total reward = -13   & ep length = 13

The agent goes from wandering aimlessly (length 252, reward −748) to navigating near-optimally (length ~13, reward ~−13) by episode 500.


📁 Project Structure
📦 sarsa-cliffwalking/
├── 📓 SARSA.ipynb          # Main notebook — full implementation
└── 📄 README.md            # You are here

🚀 Getting Started
Prerequisites

Python 3.10+
pip

Installation
bash# Clone the repository
git clone https://github.com/your-username/sarsa-cliffwalking.git
cd sarsa-cliffwalking

# Install dependencies
pip install gymnasium
pip install "gymnasium[toy-text]"
pip install numpy
Run the Notebook
bashjupyter notebook SARSA.ipynb
Or run in any Jupyter-compatible environment (VS Code, Google Colab, etc.).

🎛️ Hyperparameters
ParameterValueRolegamma (γ)0.99High discount → agent values long-term rewardsalpha (α)0.50Moderate learning rate — stable convergenceepsilon (ε)0.1010% random exploration throughout trainingepisodes500Sufficient for convergence on this environment
Experiment ideas:

Reduce epsilon over time (ε-decay) for a fully greedy final policy
Compare alpha=0.1 vs alpha=0.9 — observe stability vs. speed tradeoff
Run side-by-side with Q-Learning to visualize the path difference


💡 Key Takeaways

On-policy learning is inherently risk-aware. Because SARSA evaluates the policy it actually follows (including exploration steps), it accounts for the possibility of taking random actions near the cliff — and learns to avoid them.
The Q-table is sufficient for tabular MDPs. For environments with small, discrete state-action spaces, a plain NumPy array outperforms neural networks in speed, interpretability, and sample efficiency.
TD methods don't need complete episodes. Unlike Monte Carlo, SARSA updates at every step using only one-step lookahead — enabling learning in continuing tasks and faster propagation of rewards.
SARSA converges to the optimal ε-soft policy, not the true optimal — a subtle but important distinction from Q-Learning.


📚 Further Reading

Sutton & Barto — Reinforcement Learning: An Introduction (Ch. 6) — the canonical RL textbook
Gymnasium CliffWalking Docs — official environment reference
SARSA vs Q-Learning visual comparison — Wikipedia walkthrough



Built with curiosity and the Bellman equation.
If this helped you understand SARSA, consider leaving a ⭐

