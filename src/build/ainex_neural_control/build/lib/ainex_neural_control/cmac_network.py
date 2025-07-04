#!/usr/bin/env python3
"""
CMAC Neural Network Implementation
"""

import numpy as np

class CMAC:
    def __init__(self, input_dims=2, output_dims=2, resolution=50, receptive_field=3):
        """
        Initialize CMAC network
        """
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
        
    def _normalize_input(self, input_data):
        """Normalize input to [0, 1] range"""
        if self.input_min is None or self.input_max is None:
            self.input_min = np.min(input_data, axis=0)
            self.input_max = np.max(input_data, axis=0)
        
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
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: MSE = {mse:.6f}")
        
        print(f"Training completed! Final MSE = {self.training_errors[-1]:.6f}")
    
    def evaluate(self, input_data, target_data):
        """Evaluate model performance"""
        from sklearn.metrics import mean_squared_error
        predictions = self.predict(input_data)
        mse = mean_squared_error(target_data, predictions)
        return mse, predictions