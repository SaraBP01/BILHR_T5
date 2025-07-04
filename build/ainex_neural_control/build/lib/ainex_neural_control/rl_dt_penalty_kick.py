#!/usr/bin/env python3
"""
RL-DT Algorithm Implementation for Penalty Kicks
File: src/ainex_neural_control/rl_dt_penalty_kick.py
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import matplotlib.pyplot as plt
from collections import defaultdict
import random

class RLDT:
    def __init__(self, max_reward=20, discount_factor=0.9, max_steps=5):
        # Algorithm parameters
        self.max_reward = max_reward  # RMax from paper
        self.gamma = discount_factor  # Discount factor
        self.max_steps = max_steps    # maxsteps from paper
        
        # States and actions
        self.actions = ['MOVE_LEFT', 'MOVE_RIGHT', 'KICK']
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        
        # Transition and reward model (Decision Trees)
        self.transition_trees = {}  # One tree per state feature
        self.reward_tree = DecisionTreeRegressor(min_samples_split=2, min_samples_leaf=1)
        
        # Counters and experiences
        self.visit_counts = defaultdict(int)  # visits(s,a)
        self.experiences = []  # List of (s, a, r, s') 
        self.states_seen = set()
        
        # Value function and policy
        self.Q_values = defaultdict(float)
        self.exploration_mode = True
        self.episode_rewards = []
        
        # For plotting
        self.cumulative_rewards = []
        self.episode_count = 0

    def discretize_state(self, marker_x, leg_distance):
        """
        Discretizes continuous state into bins as in the paper.
        marker_x: x coordinate of ArUco marker in pixels
        leg_distance: foot distance in mm
        """
        # Discretize marker x coordinate (2 pixel bins)
        marker_x_discrete = int(marker_x // 2)
        
        # Discretize leg distance (4mm bins)
        leg_distance_discrete = int(leg_distance // 4)
        
        return (marker_x_discrete, leg_distance_discrete)

    def get_state_action_key(self, state, action):
        """Creates unique key for state-action pair"""
        return (state, action)

    def add_experience(self, state, action, reward, next_state):
        """Adds experience and updates model"""
        # Increment visit counter
        sa_key = self.get_state_action_key(state, action)
        self.visit_counts[sa_key] += 1
        
        # Add to experiences
        self.experiences.append((state, action, reward, next_state))
        self.states_seen.add(state)
        self.states_seen.add(next_state)
        
        # Update model with decision trees
        self.update_model()
        
        # Check if switching exploration/exploitation mode
        self.check_policy()

    def update_model(self):
        """Updates decision trees with experiences"""
        if len(self.experiences) < 2:
            return
            
        # Prepare data for training
        X = []  # [state_features + action]
        y_transitions = [[], []]  # Relative changes for each state feature
        y_rewards = []
        
        for state, action, reward, next_state in self.experiences:
            # Input: current state + action
            state_features = list(state)
            action_encoded = [0, 0, 0]
            action_encoded[self.action_to_idx[action]] = 1
            x_input = state_features + action_encoded
            X.append(x_input)
            
            # Output: relative changes in state
            for i in range(len(state)):
                relative_change = next_state[i] - state[i]
                y_transitions[i].append(relative_change)
            
            # Output: reward
            y_rewards.append(reward)
        
        X = np.array(X)
        
        # Train transition trees (one per state feature)
        for i in range(len(state)):
            if f'transition_tree_{i}' not in self.transition_trees:
                self.transition_trees[f'transition_tree_{i}'] = DecisionTreeRegressor(
                    min_samples_split=2, min_samples_leaf=1)
            
            y_trans = np.array(y_transitions[i])
            if len(np.unique(y_trans)) > 1:  # Only train if there's variability
                self.transition_trees[f'transition_tree_{i}'].fit(X, y_trans)
        
        # Train reward tree
        y_rewards = np.array(y_rewards)
        if len(np.unique(y_rewards)) > 1:
            self.reward_tree.fit(X, y_rewards)

    def predict_transition_and_reward(self, state, action):
        """Predicts next state and reward using decision trees"""
        # Prepare input
        state_features = list(state)
        action_encoded = [0, 0, 0]
        action_encoded[self.action_to_idx[action]] = 1
        x_input = np.array([state_features + action_encoded])
        
        # Predict relative changes
        predicted_state = list(state)
        for i in range(len(state)):
            tree_key = f'transition_tree_{i}'
            if tree_key in self.transition_trees and hasattr(self.transition_trees[tree_key], 'predict'):
                try:
                    relative_change = self.transition_trees[tree_key].predict(x_input)[0]
                    predicted_state[i] = state[i] + relative_change
                except:
                    pass  # Keep original state if error
        
        # Predict reward
        try:
            predicted_reward = self.reward_tree.predict(x_input)[0]
        except:
            predicted_reward = 0
        
        return tuple(predicted_state), predicted_reward

    def check_policy(self):
        """
        Checks whether to switch between exploration and exploitation.
        Explores if predicted rewards are < 40% of maximum.
        """
        if len(self.experiences) < 5:
            self.exploration_mode = True
            return
        
        # Calculate average reward of recent episodes
        recent_rewards = self.episode_rewards[-3:] if len(self.episode_rewards) >= 3 else self.episode_rewards
        if recent_rewards:
            avg_reward = np.mean(recent_rewards)
            threshold = 0.4 * self.max_reward
            
            if avg_reward < threshold:
                self.exploration_mode = True
            else:
                self.exploration_mode = False

    def select_action(self, state):
        """Selects action using epsilon-greedy or directed exploration"""
        if self.exploration_mode:
            # Directed exploration: favor less visited actions
            action_visits = []
            for action in self.actions:
                sa_key = self.get_state_action_key(state, action)
                visits = self.visit_counts[sa_key]
                action_visits.append((action, visits))
            
            # Select action with fewest visits
            min_visits = min(action_visits, key=lambda x: x[1])[1]
            least_visited = [a for a, v in action_visits if v == min_visits]
            return random.choice(least_visited)
        else:
            # Exploitation: use Q-values
            q_values = []
            for action in self.actions:
                q_key = self.get_state_action_key(state, action)
                q_values.append((action, self.Q_values[q_key]))
            
            return max(q_values, key=lambda x: x[1])[0]

    def value_iteration(self):
        """Performs value iteration on learned model"""
        if len(self.states_seen) == 0:
            return
        
        # Value iteration convergence
        max_iterations = 100
        tolerance = 1e-4
        
        for iteration in range(max_iterations):
            old_q_values = self.Q_values.copy()
            
            for state in self.states_seen:
                for action in self.actions:
                    sa_key = self.get_state_action_key(state, action)
                    
                    if self.exploration_mode:
                        # In exploration mode, give bonus to less visited states
                        visits = self.visit_counts[sa_key]
                        min_visits = min([self.visit_counts[self.get_state_action_key(s, a)] 
                                        for s in self.states_seen for a in self.actions])
                        
                        if visits == min_visits:
                            self.Q_values[sa_key] = self.max_reward
                        else:
                            # Use learned model
                            next_state, reward = self.predict_transition_and_reward(state, action)
                            
                            # Calculate Q-value
                            max_q_next = max([self.Q_values[self.get_state_action_key(next_state, a)] 
                                            for a in self.actions])
                            self.Q_values[sa_key] = reward + self.gamma * max_q_next
                    else:
                        # In exploitation mode, use only the model
                        next_state, reward = self.predict_transition_and_reward(state, action)
                        max_q_next = max([self.Q_values[self.get_state_action_key(next_state, a)] 
                                        for a in self.actions])
                        self.Q_values[sa_key] = reward + self.gamma * max_q_next
            
            # Check convergence
            converged = all(abs(self.Q_values[key] - old_q_values.get(key, 0)) < tolerance 
                          for key in self.Q_values)
            if converged:
                break

    def plot_cumulative_reward(self):
        """Plots cumulative reward"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.cumulative_rewards)), self.cumulative_rewards, 'b-', linewidth=2)
        plt.xlabel('Episode Number')
        plt.ylabel('Cumulative Reward')
        plt.title('RL-DT Learning Progress: Cumulative Reward vs Episodes')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return plt.gcf()

# Example usage for testing
def run_penalty_kick_experiment():
    """Runs complete penalty kick experiment"""
    
    # Create RL-DT agent
    agent = RLDT(max_reward=20, discount_factor=0.9)
    
    # Run training episodes
    num_episodes = 100
    print("Starting RL-DT training for penalty kicks...")
    
    for episode in range(num_episodes):
        # Simulate episode (this would be replaced by real robot interaction)
        episode_reward = simulate_episode(agent)
        
        if episode % 10 == 0:
            mode = "Exploration" if agent.exploration_mode else "Exploitation"
            print(f"Episode {episode}: Reward = {episode_reward:.1f}, Mode = {mode}")
    
    # Show final results
    print(f"\nFinal results:")
    print(f"Total episodes: {num_episodes}")
    print(f"Total cumulative reward: {agent.cumulative_rewards[-1]:.1f}")
    print(f"Average reward per episode: {agent.cumulative_rewards[-1]/num_episodes:.2f}")
    print(f"States explored: {len(agent.states_seen)}")
    print(f"Experiences collected: {len(agent.experiences)}")
    
    # Show learned policy
    print(f"\nLearned policy (last visited states):")
    recent_states = list(agent.states_seen)[-5:]  # Show last 5 states
    for state in recent_states:
        best_action = agent.select_action(state)
        print(f"State {state} -> Action: {best_action}")
    
    # Plot progress
    agent.plot_cumulative_reward()
    
    return agent

def simulate_episode(agent):
    """Simple episode simulation for testing"""
    # Initial state: marker centered, leg in neutral position
    current_state = agent.discretize_state(marker_x=160, leg_distance=100)
    episode_reward = 0
    steps = 0
    max_steps = 20
    
    while steps < max_steps:
        # Select action
        action = agent.select_action(current_state)
        
        # Simulate environment response
        next_state, reward = simulate_environment_response(current_state, action, agent)
        
        # Add experience
        agent.add_experience(current_state, action, reward, next_state)
        
        episode_reward += reward
        current_state = next_state
        steps += 1
        
        # End if KICK was executed
        if action == 'KICK':
            break
    
    # Update value function
    agent.value_iteration()
    
    # Save episode reward
    agent.episode_rewards.append(episode_reward)
    agent.cumulative_rewards.append(sum(agent.episode_rewards))
    agent.episode_count += 1
    
    return episode_reward

def simulate_environment_response(state, action, agent):
    """Simulates environment response for testing"""
    marker_x, leg_distance = state
    
    if action == 'MOVE_LEFT':
        # Move leg inward (reduce distance)
        new_leg_distance = max(0, leg_distance - 1)  # -4mm discretized
        reward = -1  # Movement penalty
        next_state = agent.discretize_state(marker_x * 2, new_leg_distance * 4)
        
    elif action == 'MOVE_RIGHT':
        # Move leg outward (increase distance)
        new_leg_distance = min(50, leg_distance + 1)  # +4mm discretized
        reward = -1  # Movement penalty
        next_state = agent.discretize_state(marker_x * 2, new_leg_distance * 4)
        
    elif action == 'KICK':
        # Evaluate shot success based on marker alignment
        # Target: center of image (assuming camera is centered)
        image_center = 80  # 160 pixels // 2 (discretized)
        distance_from_center = abs(marker_x - image_center)
        
        # Success if marker is close to center
        if distance_from_center < 10:  # 20 pixel tolerance (discretized)
            reward = 20  # Hit target
        else:
            reward = -2  # Miss target
            
        next_state = state  # State doesn't change after action
    
    else:
        next_state = state
        reward = 0
    
    return next_state, reward

if __name__ == "__main__":
    # Run experiment
    trained_agent = run_penalty_kick_experiment()