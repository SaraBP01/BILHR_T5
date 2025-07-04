#!/usr/bin/env python3
"""
CMAC Neural Network Implementation for Robot Arm Control - COMPLETE VERSION
Tutorial 4 - BILHR SS 2025

This version includes all plotting functions and debugging.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import sys
from cmac_network import CMAC  # Assuming CMAC class is in cmac_network.py

class CMAC:
    def __init__(self, input_dims=2, output_dims=2, resolution=50, receptive_field=3):
        """
        Initialize CMAC network
        """
        print(f"DEBUG: Initializing CMAC with resolution={resolution}, RF={receptive_field}")
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.resolution = resolution
        self.receptive_field = receptive_field
        
        # Calculate total number of memory cells
        self.total_cells = (resolution ** input_dims) * receptive_field
        
        # Initialize weight matrix
        self.weights = np.random.normal(0, 0.1, (self.total_cells, output_dims))
        
        # Keep track of input ranges for normalization
        self.input_min = None
        self.input_max = None
        
        # Training statistics
        self.training_errors = []
        
        print(f"DEBUG: CMAC initialized with {self.total_cells} memory cells")
        
    def _normalize_input(self, input_data):
        """Normalize input to [0, 1] range"""
        if self.input_min is None or self.input_max is None:
            self.input_min = np.min(input_data, axis=0)
            self.input_max = np.max(input_data, axis=0)
            print(f"DEBUG: Input ranges - Min: {self.input_min}, Max: {self.input_max}")
        
        # Avoid division by zero
        range_vals = self.input_max - self.input_min
        range_vals[range_vals == 0] = 1
        
        normalized = (input_data - self.input_min) / range_vals
        return np.clip(normalized, 0, 0.999)
    
    def _get_active_cells(self, normalized_input):
        """Get indices of active memory cells for given input"""
        active_cells = []
        
        for offset in range(self.receptive_field):
            grid_pos = []
            for dim in range(self.input_dims):
                pos = (normalized_input[dim] * self.resolution + 
                      offset * self.resolution / self.receptive_field) % self.resolution
                grid_pos.append(int(pos))
            
            cell_index = 0
            for dim in range(self.input_dims):
                cell_index += grid_pos[dim] * (self.resolution ** dim)
            
            cell_index += offset * (self.resolution ** self.input_dims)
            active_cells.append(cell_index)
            
        return active_cells
    
    def predict(self, input_data):
        """Predict output for given input"""
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
            
        normalized_inputs = self._normalize_input(input_data)
        predictions = []
        
        for inp in normalized_inputs:
            active_cells = self._get_active_cells(inp)
            output = np.mean(self.weights[active_cells], axis=0)
            predictions.append(output)
            
        return np.array(predictions)
    
    def train(self, input_data, target_data, learning_rate=0.1, epochs=100):
        """Train the CMAC network"""
        print(f"DEBUG: Starting training with {len(input_data)} samples for {epochs} epochs...")
        print(f"DEBUG: Learning rate = {learning_rate}")
        
        # Force flush output
        sys.stdout.flush()
        
        # Normalize inputs
        normalized_inputs = self._normalize_input(input_data)
        
        for epoch in range(epochs):
            epoch_error = 0
            
            for i, (inp, target) in enumerate(zip(normalized_inputs, target_data)):
                # Get active cells for this input
                active_cells = self._get_active_cells(inp)
                
                # Forward pass
                prediction = np.mean(self.weights[active_cells], axis=0)
                
                # Calculate error
                error = target - prediction
                epoch_error += np.sum(error ** 2)
                
                # Update weights of active cells
                for cell_idx in active_cells:
                    self.weights[cell_idx] += learning_rate * error / len(active_cells)
            
            # Store training error
            mse = epoch_error / len(input_data)
            self.training_errors.append(mse)
            
            # Print more frequently and force flush
            if epoch % 10 == 0:
                print(f"DEBUG: Epoch {epoch}: MSE = {mse:.6f}")
                sys.stdout.flush()
        
        print(f"DEBUG: Training completed! Final MSE = {self.training_errors[-1]:.6f}")
        sys.stdout.flush()
    
    def evaluate(self, input_data, target_data):
        """Evaluate model performance"""
        predictions = self.predict(input_data)
        mse = mean_squared_error(target_data, predictions)
        return mse, predictions


def load_training_data(filename):
    """Load training data from pickle file"""
    print(f"DEBUG: Attempting to load data from: {filename}")
    sys.stdout.flush()
    
    try:
        if not os.path.exists(filename):
            print(f"DEBUG: File does not exist: {filename}")
            return None
            
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"DEBUG: Successfully loaded {len(data)} training samples")
        sys.stdout.flush()
        return data
    except FileNotFoundError:
        print(f"DEBUG: FileNotFoundError for {filename}")
        return None
    except Exception as e:
        print(f"DEBUG: Exception loading data: {e}")
        return None


def prepare_data(raw_data):
    """Convert raw data to input/output arrays"""
    if not raw_data:
        print("DEBUG: No raw data provided")
        return None, None
    
    print(f"DEBUG: Preparing data - raw data length: {len(raw_data)}")
    print(f"DEBUG: First sample: {raw_data[0]}")
    sys.stdout.flush()
    
    data_array = np.array(raw_data)
    
    # Split into inputs and outputs
    inputs = data_array[:, :2]
    outputs = data_array[:, 2:]
    
    print(f"DEBUG: Input shape: {inputs.shape}, Output shape: {outputs.shape}")
    print(f"DEBUG: Input range: X=[{inputs[:, 0].min():.2f}, {inputs[:, 0].max():.2f}], "
          f"Y=[{inputs[:, 1].min():.2f}, {inputs[:, 1].max():.2f}]")
    print(f"DEBUG: Output range: Roll=[{outputs[:, 0].min():.4f}, {outputs[:, 0].max():.4f}], "
          f"Pitch=[{outputs[:, 1].min():.4f}, {outputs[:, 1].max():.4f}]")
    sys.stdout.flush()
    
    return inputs, outputs


def plot_training_results(results):
    """Plot training results and MSE over epochs"""
    print("DEBUG: Creating training results plot...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training errors over epochs
    axes[0, 0].plot(results['case_a']['cmac'].training_errors, label='Case A (75 samples)', color='blue')
    axes[0, 1].plot(results['case_b']['cmac'].training_errors, label='Case B (150 samples)', color='green')
    axes[0, 0].set_title('Case A: Training Error Over Epochs')
    axes[0, 1].set_title('Case B: Training Error Over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 0].set_yscale('log')  # Log scale for better visualization
    axes[0, 1].set_yscale('log')
    axes[0, 0].grid(True)
    axes[0, 1].grid(True)
    
    # Plot predictions vs targets for shoulder roll
    for i, (case_name, case_data) in enumerate(results.items()):
        ax = axes[1, i]
        targets = case_data['targets'][:, 0]  # shoulder roll
        predictions = case_data['predictions'][:, 0]
        
        ax.scatter(targets, predictions, alpha=0.6)
        ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        ax.set_xlabel('Target Shoulder Roll')
        ax.set_ylabel('Predicted Shoulder Roll')
        ax.set_title(f'{case_name.upper()}: Shoulder Roll Prediction')
        ax.grid(True)
    
    plt.tight_layout()
    plt.show(block=False)  # Non-blocking show
    plt.pause(1)  # Pause to render
    print("DEBUG: Training results plot displayed (close to continue)")


def plot_rf5_comparison(case_b_cmac, rf5_cmac, inputs, outputs):
    """Plot comparison between RF=3 and RF=5"""
    print("DEBUG: Creating RF comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training errors comparison
    axes[0, 0].plot(case_b_cmac.training_errors, label='RF=3 (150 samples)', color='blue')
    axes[0, 1].plot(rf5_cmac.training_errors, label='RF=5 (150 samples)', color='red')
    axes[0, 0].set_title('RF=3: Training Error Over Epochs')
    axes[0, 1].set_title('RF=5: Training Error Over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 0].set_yscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 0].grid(True)
    axes[0, 1].grid(True)
    
    # Predictions comparison for shoulder roll
    rf3_pred = case_b_cmac.predict(inputs)
    rf5_pred = rf5_cmac.predict(inputs)
    
    axes[1, 0].scatter(outputs[:, 0], rf3_pred[:, 0], alpha=0.6, color='blue')
    axes[1, 0].plot([outputs[:, 0].min(), outputs[:, 0].max()], 
                    [outputs[:, 0].min(), outputs[:, 0].max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Target Shoulder Roll')
    axes[1, 0].set_ylabel('Predicted Shoulder Roll')
    axes[1, 0].set_title('RF=3: Shoulder Roll Prediction')
    axes[1, 0].grid(True)
    
    axes[1, 1].scatter(outputs[:, 0], rf5_pred[:, 0], alpha=0.6, color='red')
    axes[1, 1].plot([outputs[:, 0].min(), outputs[:, 0].max()], 
                    [outputs[:, 0].min(), outputs[:, 0].max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('Target Shoulder Roll')
    axes[1, 1].set_ylabel('Predicted Shoulder Roll')
    axes[1, 1].set_title('RF=5: Shoulder Roll Prediction')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    print("DEBUG: RF comparison plot displayed")


def main():
    """Main function with extensive debugging"""
    print("DEBUG: Starting CMAC training program")
    print("DEBUG: Python version:", sys.version)
    print("DEBUG: Current working directory:", os.getcwd())
    sys.stdout.flush()
    
    # List all possible file paths to try
    possible_paths = [
        'src/ainex_vision/ainex_vision/data_collected/blob_shoulder_data_T4.pkl',
        'blob_shoulder_data_T4.pkl',
        'data_collected/blob_shoulder_data_T4.pkl',
        '../data_collected/blob_shoulder_data_T4.pkl',
        './blob_shoulder_data_T4.pkl'
    ]
    
    print("DEBUG: Checking for data files...")
    for path in possible_paths:
        exists = os.path.exists(path)
        print(f"DEBUG: {path} - {'EXISTS' if exists else 'NOT FOUND'}")
    sys.stdout.flush()
    
    # Try to load data
    raw_data = None
    for path in possible_paths:
        print(f"DEBUG: Trying to load from: {path}")
        raw_data = load_training_data(path)
        if raw_data is not None:
            print(f"DEBUG: Successfully loaded data from: {path}")
            break
    
    if raw_data is None:
        print("DEBUG: ERROR - Could not find training data file!")
        print("DEBUG: Please make sure the file 'blob_shoulder_data_T4.pkl' is in one of these locations:")
        for path in possible_paths:
            print(f"DEBUG:   - {path}")
        return None
    
    # Prepare data
    print("DEBUG: Preparing data...")
    sys.stdout.flush()
    inputs, outputs = prepare_data(raw_data)
    if inputs is None:
        print("DEBUG: ERROR - Could not prepare data!")
        return None
    
    # Case A: 75 samples
    print("\n" + "="*60)
    print("DEBUG: CASE A - Training with 75 samples")
    print("="*60)
    sys.stdout.flush()
    
    if len(inputs) >= 75:
        train_inputs_a = inputs[:75]
        train_outputs_a = outputs[:75]
        
        print(f"DEBUG: Case A - Training data shape: {train_inputs_a.shape}")
        sys.stdout.flush()
        
        cmac_a = CMAC(input_dims=2, output_dims=2, resolution=50, receptive_field=3)
        cmac_a.train(train_inputs_a, train_outputs_a, learning_rate=0.1, epochs=100)
        
        mse_a, pred_a = cmac_a.evaluate(train_inputs_a, train_outputs_a)
        print(f"DEBUG: Case A - Final Training MSE: {mse_a:.6f}")
        sys.stdout.flush()
    else:
        print(f"DEBUG: ERROR - Not enough samples! Have {len(inputs)}, need 75")
        return None
    
    # Case B: 150 samples
    print("\n" + "="*60)
    print(f"DEBUG: CASE B - Training with {min(150, len(inputs))} samples")
    print("="*60)
    sys.stdout.flush()
    
    n_samples_b = min(150, len(inputs))
    train_inputs_b = inputs[:n_samples_b]
    train_outputs_b = outputs[:n_samples_b]
    
    print(f"DEBUG: Case B - Training data shape: {train_inputs_b.shape}")
    sys.stdout.flush()
    
    cmac_b = CMAC(input_dims=2, output_dims=2, resolution=50, receptive_field=3)
    cmac_b.train(train_inputs_b, train_outputs_b, learning_rate=0.1, epochs=100)
    
    mse_b, pred_b = cmac_b.evaluate(train_inputs_b, train_outputs_b)
    print(f"DEBUG: Case B - Final Training MSE: {mse_b:.6f}")
    sys.stdout.flush()
    
    # Create results dictionary for plotting
    results = {
        'case_a': {
            'cmac': cmac_a,
            'mse': mse_a,
            'predictions': pred_a,
            'inputs': train_inputs_a,
            'targets': train_outputs_a
        },
        'case_b': {
            'cmac': cmac_b,
            'mse': mse_b,
            'predictions': pred_b,
            'inputs': train_inputs_b,
            'targets': train_outputs_b
        }
    }
    
    # Plot results for Case A and B
    print("DEBUG: Generating plots for Cases A and B...")
    plot_training_results(results)
    
    # Test with receptive field = 5
    print("\n" + "="*60)
    print("DEBUG: TESTING - Receptive field = 5")
    print("="*60)
    sys.stdout.flush()
    
    cmac_rf5 = CMAC(input_dims=2, output_dims=2, resolution=50, receptive_field=5)
    cmac_rf5.train(train_inputs_b, train_outputs_b, learning_rate=0.1, epochs=100)
    
    mse_rf5, pred_rf5 = cmac_rf5.evaluate(train_inputs_b, train_outputs_b)
    print(f"DEBUG: Receptive field = 5 - Final Training MSE: {mse_rf5:.6f}")
    sys.stdout.flush()
    
    # Plot comparison between RF=3 and RF=5
    print("DEBUG: Generating comparison plots for RF=3 vs RF=5...")
    plot_rf5_comparison(cmac_b, cmac_rf5, train_inputs_b, train_outputs_b)
    
    # Final summary
    print("\n" + "="*60)
    print("DEBUG: FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Case A (75 samples):     MSE = {mse_a:.6f}")
    print(f"Case B (150 samples):    MSE = {mse_b:.6f}")
    print(f"Receptive field = 5:     MSE = {mse_rf5:.6f}")
    print("="*60)
    sys.stdout.flush()
    
    # Try to save models
    try:
        with open('cmac_case_a.pkl', 'wb') as f:
            pickle.dump(cmac_a, f)
        with open('cmac_case_b.pkl', 'wb') as f:
            pickle.dump(cmac_b, f)
        with open('cmac_rf5.pkl', 'wb') as f:
            pickle.dump(cmac_rf5, f)
        print("DEBUG: Models saved successfully!")
    except Exception as e:
        print(f"DEBUG: Error saving models: {e}")
    
    sys.stdout.flush()
    
    return {
        'case_a_mse': mse_a,
        'case_b_mse': mse_b,
        'rf5_mse': mse_rf5
    }


if __name__ == '__main__':
    print("DEBUG: Script starting...")
    sys.stdout.flush()
    results = main()
    print("DEBUG: Script completed!")
    sys.stdout.flush()