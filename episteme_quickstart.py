"""
Episteme Framework - Quick Start Example

Demonstrates the full active learning loop on a simple 1D ecosystem.
"""

import numpy as np
import matplotlib.pyplot as plt
from episteme import EpistemeAgent, EpistemeConfig


# ============================================================================
# 1. Define the Environment (True System - Unknown to Agent)
# ============================================================================

class LogisticEnvironment:
    """
    1D population dynamics with logistic growth:
        dN/dt = r*N*(1 - N/K) + u + noise
    
    Parameters:
        r: growth rate (ground truth: 0.8)
        K: carrying capacity (ground truth: 50.0)
    """
    
    def __init__(self, r=0.8, K=50.0, dt=1.0, noise=0.5):
        self.r = r
        self.K = K
        self.dt = dt
        self.noise = noise
        self.state = 25.0  # Initial population
        self.time = 0.0
        
    def step(self, action):
        """
        Execute action and return noisy observation.
        
        Args:
            action: Control input (e.g., resource addition)
            
        Returns:
            Noisy observation of population
        """
        # True dynamics
        dN = self.r * self.state * (1 - self.state/self.K) + 0.5*action
        self.state = self.state + self.dt * dN
        self.state = max(0.1, self.state)  # Prevent extinction
        
        # Add observation noise
        observation = self.state + np.random.normal(0, self.noise)
        self.time += self.dt
        
        return observation
    
    def reset(self):
        """Reset to initial state"""
        self.state = 25.0
        self.time = 0.0


# ============================================================================
# 2. Configure and Initialize Agent
# ============================================================================

config = EpistemeConfig(
    inference_method="sbi",           # Use simulation-based inference
    n_simulations=1000,               # Simulations per round
    n_candidate_actions=5,            # Actions to evaluate for BOED
    action_bounds=(-5.0, 5.0),        # Action space
    symbolic_library=['x', 'x**2', 'a', 'x*a'],  # Basis functions
    use_babelfish=False,              # Disable for simple demo
)

agent = EpistemeAgent(config)


# ============================================================================
# 3. Define Agent's Generative Model (Simulator)
# ============================================================================

def simulator(theta, action):
    """
    Agent's model of the environment.
    
    Args:
        theta: Parameters [r, K]
        action: Control input
        
    Returns:
        Simulated observation
    """
    r, K = theta[0].item(), theta[1].item()
    
    # Assume agent starts near truth
    N = 25.0
    
    # Simulate for T steps
    T = 20
    dt = 1.0
    
    trajectory = []
    for t in range(T):
        dN = r * N * (1 - N/K) + 0.5*action
        N = N + dt * dN
        N = max(0.1, N)
        trajectory.append(N)
    
    # Return final state with noise
    import torch
    return torch.tensor(trajectory[-1] + np.random.normal(0, 0.5), dtype=torch.float32)


# ============================================================================
# 4. Setup Agent with Prior and Simulator
# ============================================================================

import torch

# Define prior over parameters [r, K]
prior = torch.distributions.Uniform(
    low=torch.tensor([0.1, 30.0]),
    high=torch.tensor([2.0, 70.0])
)

# Give agent the simulator
agent.sbi_engine.setup(prior, simulator)


# ============================================================================
# 5. Run Active Learning Loop
# ============================================================================

env = LogisticEnvironment(r=0.8, K=50.0)

n_rounds = 10
results = {
    'actions': [],
    'observations': [],
    'true_states': [],
    'r_samples': [],
    'K_samples': [],
}

print("Starting Episteme Active Learning Loop...")
print("=" * 60)

for round_idx in range(n_rounds):
    # 1. DESIGN: Agent selects action via BOED
    if round_idx == 0:
        action = 0.0  # Start with neutral action
    else:
        action = agent.select_action_boed(n_samples=100)
    
    # 2. ACT: Execute in true environment
    observation = env.step(action)
    
    # 3. LEARN: Update posterior
    agent.update_posterior(torch.tensor([observation]), action)
    
    # Store results
    results['actions'].append(action)
    results['observations'].append(observation)
    results['true_states'].append(env.state)
    
    # Sample from posterior
    if agent.posterior is not None:
        samples = agent.posterior.sample((500,))
        results['r_samples'].append(samples[:, 0].numpy())
        results['K_samples'].append(samples[:, 1].numpy())
    
    # 4. DISCOVER: Symbolic regression (every 3 rounds)
    if round_idx % 3 == 0 and round_idx > 0:
        # Prepare data for symbolic regression
        X = np.array(results['observations'][:-1]).reshape(-1, 1)
        actions_array = np.array(results['actions'])
        y = np.diff(results['observations'])
        
        equations = agent.symbolic_regressor.fit(X, y, actions_array[:-1])
        print(f"\nRound {round_idx} - Discovered Equation:")
        print(f"  {equations['equation']}")
        print(f"  R² = {equations['r2_score']:.3f}")
    
    # Print progress
    if agent.posterior is not None:
        r_mean = samples[:, 0].mean().item()
        K_mean = samples[:, 1].mean().item()
        print(f"Round {round_idx:2d} | Action: {action:+.2f} | "
              f"r: {r_mean:.3f} | K: {K_mean:.2f}")


# ============================================================================
# 6. Visualize Results
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Population trajectory
ax = axes[0, 0]
ax.plot(results['true_states'], 'o-', label='True Population', linewidth=2)
ax.plot(results['observations'], 's--', alpha=0.6, label='Observations')
ax.set_xlabel('Round')
ax.set_ylabel('Population')
ax.set_title('Population Dynamics')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Actions over time
ax = axes[0, 1]
ax.plot(results['actions'], 'o-', color='green', linewidth=2)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Round')
ax.set_ylabel('Action')
ax.set_title('Selected Actions (BOED)')
ax.grid(True, alpha=0.3)

# Plot 3: Parameter learning - r
ax = axes[1, 0]
for i, samples in enumerate(results['r_samples']):
    ax.violinplot([samples], positions=[i], widths=0.7, 
                   showmeans=True, showextrema=False)
ax.axhline(y=0.8, color='r', linestyle='--', linewidth=2, label='True r')
ax.set_xlabel('Round')
ax.set_ylabel('Growth Rate (r)')
ax.set_title('Posterior Distribution: r')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Parameter learning - K
ax = axes[1, 1]
for i, samples in enumerate(results['K_samples']):
    ax.violinplot([samples], positions=[i], widths=0.7,
                   showmeans=True, showextrema=False)
ax.axhline(y=50.0, color='r', linestyle='--', linewidth=2, label='True K')
ax.set_xlabel('Round')
ax.set_ylabel('Carrying Capacity (K)')
ax.set_title('Posterior Distribution: K')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('episteme_quickstart_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Active Learning Complete!")
print(f"Final estimates:")
print(f"  r = {results['r_samples'][-1].mean():.3f} ± {results['r_samples'][-1].std():.3f} (true: 0.8)")
print(f"  K = {results['K_samples'][-1].mean():.1f} ± {results['K_samples'][-1].std():.1f} (true: 50.0)")
print("=" * 60)
