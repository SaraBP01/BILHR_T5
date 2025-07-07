import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class RLDT:
    def __init__(self, max_reward=20, discount_factor=0.9):
        self.max_reward = max_reward
        self.gamma = discount_factor
        self.q_values = {}  # (state, action) → value
        self.visit_counts = {}
        
        # Training data
        self.transition_data = []  # (s, a) → Δstate
        self.transition_labels = []  # new state or relative leg pos
        self.reward_data = []  # (s, a) → reward
        
        # Trees
        self.transition_model = DecisionTreeRegressor()
        self.reward_model = DecisionTreeRegressor()
        self.model_trained = False

        # Tracking
        self.states = set()
        self.actions = ['MOVE_LEFT', 'MOVE_RIGHT', 'KICK']
        self.episode_rewards = []
        self.exploration_mode = True

    def discretize_state(self, marker_x, leg_pos_mm):
        # Keep it compact and hashable
        mx = int(marker_x // 5)
        lp = int(leg_pos_mm // 4)
        return f"S{mx}_{lp}"

    def add_experience(self, state, action, reward, next_state):
        # Update tracking
        self.states.add(state)
        self.visit_counts[(state, action)] = self.visit_counts.get((state, action), 0) + 1
        
        # Extract numeric features
        s_features = self._parse_state(state)
        a_index = self.actions.index(action)
        sa_vector = np.array([*s_features, a_index])

        # Save training data
        next_leg = self._parse_state(next_state)[1]
        self.transition_data.append(sa_vector)
        self.transition_labels.append(next_leg - s_features[1])
        self.reward_data.append((sa_vector, reward))

        # For tracking
        if action == "KICK":
            if len(self.episode_rewards) == 0 or len(self.episode_rewards) < len(self.states):
                self.episode_rewards.append(reward)
            else:
                self.episode_rewards[-1] += reward

    def select_action(self, state):
        # If model is not good yet, explore randomly
        if not self.model_trained or self.exploration_mode:
            return np.random.choice(self.actions)

        # Greedy action from Q
        best_action = max(self.actions, key=lambda a: self.q_values.get((state, a), 0))
        return best_action

    def value_iteration(self, iterations=10):
        # Train models
        if len(self.transition_data) < 5:
            return  # Not enough data

        X = np.array(self.transition_data)
        y_trans = np.array(self.transition_labels)
        y_rew = np.array([r for _, r in self.reward_data])
        X_rew = np.array([x for x, _ in self.reward_data])

        self.transition_model.fit(X, y_trans)
        self.reward_model.fit(X_rew, y_rew)
        self.model_trained = True

        # Value iteration loop
        for _ in range(iterations):
            for s in list(self.states):
                s_feat = self._parse_state(s)
                for a in self.actions:
                    sa_vec = np.array([*s_feat, self.actions.index(a)]).reshape(1, -1)
                    
                    # Predict next state
                    delta = self.transition_model.predict(sa_vec)[0]
                    next_leg = s_feat[1] + delta
                    next_state = self.discretize_state(s_feat[0] * 5, next_leg * 4)

                    # Predict reward
                    reward = self.reward_model.predict(sa_vec)[0]

                    # Update Q
                    max_next = max([self.q_values.get((next_state, ap), 0) for ap in self.actions])
                    self.q_values[(s, a)] = reward + self.gamma * max_next

        # Check for exploitation condition
        self.exploration_mode = not self._is_confident()

    def _is_confident(self):
        # Go to exploitation if model predicts reward ≥ 0.4 * Rmax from any state
        for s in self.states:
            s_feat = self._parse_state(s)
            for a in self.actions:
                sa_vec = np.array([*s_feat, self.actions.index(a)]).reshape(1, -1)
                if self.model_trained:
                    r = self.reward_model.predict(sa_vec)[0]
                    if r >= 0.4 * self.max_reward:
                        return True
        return False

    def _parse_state(self, state_str):
        """Convert state string back to features (int)"""
        _, mx, lp = state_str.split("_")
        return int(mx), int(lp)

    def plot_cumulative_reward(self):
        import matplotlib.pyplot as plt
        cumulative = np.cumsum(self.episode_rewards)
        fig, ax = plt.subplots()
        ax.plot(cumulative)
        ax.set_title("RL-DT Cumulative Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        return fig