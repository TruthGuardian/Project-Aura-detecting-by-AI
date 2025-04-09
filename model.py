def visualize_composition_correlation(self, df, n_particles=8, figsize=(18, 10)):
        """
        Visualize the correlation between halo detector signals and beam composition
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Data to analyze
        n_particles : int
            Number of particle types to include (most abundant)
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure : Correlation visualization
        """
        if not hasattr(self, 'comp_cols'):
            raise ValueError("Model doesn't have composition column information")
        
        # Check if we have all necessary columns
        if not all(col in df.columns for col in self.comp_cols):
            raise ValueError("Not all composition columns found in data")
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Top-left: Composition distribution
        ax = axes[0, 0]
        
        # Calculate mean composition
        comp_means = df[self.comp_cols].mean()
        
        # Sort by mean value and take top n_particles
        top_particles = comp_means.sort_values(ascending=False).head(n_particles)
        
        # Convert to DataFrame for plotting
        particle_names = [name.replace('comp_', '') for name in top_particles.index]
        
        # Create bar chart
        ax.bar(particle_names, top_particles.values, color='teal', alpha=0.7)
        ax.set_title('Average Beam Composition')
        ax.set_ylabel('Fraction')
        ax.set_ylim(0, top_particles.values[0] * 1.2)  # Scale y-axis based on highest value
        
        # Rotate x-tick labels if many particles
        if n_particles > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Top-right: Halo-Composition Correlation Matrix
        ax = axes[0, 1]
        
        # Define columns to analyze
        halo_cols = [col for col in df.columns if col.startswith('halo_')]
        comp_cols = self.comp_cols
        
        if halo_cols:
            # Calculate correlation matrix
            corr_cols = halo_cols + comp_cols
            corr_matrix = df[corr_cols].corr()
            
            # Extract the quadrant that shows correlation between halo features and composition
            halo_comp_corr = corr_matrix.loc[halo_cols, comp_cols]
            
            # Create heatmap
            sns.heatmap(halo_comp_corr, cmap='coolwarm', center=0, annot=True, 
                      fmt='.2f', linewidths=0.5, ax=ax)
            
            ax.set_title('Correlation: Halo Features vs Composition')
            ax.set_xticklabels([name.replace('comp_', '') for name in comp_cols], rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, "No halo detector data found", ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        # 3. Bottom-left: Halo Detector Response by Particle Type
        ax = axes[1, 0]
        
        # Check for ADC columns
        adc_cols = [col for col in df.columns if col.startswith('halo_') and col.endswith('_adc')]
        
        if adc_cols:
            # For each particle type, show average ADC response
            # First, create a derived column with dominant particle type
            top_comp_cols = comp_means.sort_values(ascending=False).head(5).index
            
            # Set a threshold to identify "pure" beams (where one particle dominates)
            threshold = 0.5
            
            # Calculate average ADC values for each quadrant, grouped by dominant particle
            adc_by_particle = {}
            
            for col in top_comp_cols:
                particle = col.replace('comp_', '')
                # Select events where this particle dominates
                dominant_mask = df[col] > threshold
                if dominant_mask.sum() > 5:  # Only if we have enough samples
                    adc_values = df.loc[dominant_mask, adc_cols].mean()
                    adc_by_particle[particle] = adc_values.values
            
            if adc_by_particle:
                # Convert to array for heatmap
                particles = list(adc_by_particle.keys())
                quadrants = [col.replace('halo_', '').replace('_adc', '') for col in adc_cols]
                
                adc_array = np.array([adc_by_particle[p] for p in particles])
                
                # Create heatmap
                im = ax.imshow(adc_array, cmap='viridis', aspect='auto')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Average ADC Value')
                
                # Add labels
                ax.set_yticks(np.arange(len(particles)))
                ax.set_yticklabels(particles)
                ax.set_xticks(np.arange(len(quadrants)))
                ax.set_xticklabels(quadrants)
                
                # Add text annotations
                for i in range(len(particles)):
                    for j in range(len(quadrants)):
                        text_color = "white" if adc_array[i, j] > 100 else "black"
                        ax.text(j, i, f"{adc_array[i, j]:.1f}", ha="center", va="center", color=text_color)
                
                ax.set_title('Halo Detector Response by Particle Type')
            else:
                ax.text(0.5, 0.5, "Not enough samples with dominant particles", 
                      ha='center', va='center', transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, "No ADC data available", 
                  ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        # 4. Bottom-right: Composition variation across conditions
        ax = axes[1, 1]
        
        if 'condition' in df.columns:
            # Group by condition and calculate mean composition
            condition_comp = df.groupby('condition')[self.comp_cols].mean()
            
            # Convert column names to cleaner particle names
            condition_comp.columns = [col.replace('comp_', '') for col in condition_comp.columns]
            
            # Plot as stacked bar chart
            condition_comp.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
            ax.set_title('Beam Composition by Experimental Condition')
            ax.set_ylabel('Fraction')
            ax.set_ylim(0, 1.0)
            
            # Rotate x-tick labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add a grid
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No condition data available", 
                  ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        plt.tight_layout()
        return fig
    
def analyze_beam_composition(self, df, predictions=None, preset_names=None):
        """
        Analyze beam composition in depth with additional metrics
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Test data
        predictions : dict or None
            Prediction results from predict method, or None to run predictions
        preset_names : list or None
            List of beam composition presets to analyze separately (if in data)
            
        Returns:
        --------
        dict : Analysis results including metrics by particle type and preset
        """
        if not hasattr(self, 'comp_cols'):
            raise ValueError("Model doesn't have composition column information")
        
        # Get predictions if not provided
        if predictions is None:
            predictions = self.predict(df)
        
        if 'composition' not in predictions:
            raise ValueError("No composition predictions found")
        
        # Extract true and predicted compositions
        true_comp_cols = self.comp_cols
        pred_comp = predictions['composition']
        
        analysis = {}
        
        # 1. Calculate overall metrics
        overall_metrics = {}
        
        # Make sure we have true composition data
        if all(col in df.columns for col in true_comp_cols):
            true_comp = df[true_comp_cols].values
            
            # Overall metrics
            overall_metrics['mse'] = mean_squared_error(true_comp, pred_comp)
            overall_metrics['mae'] = mean_absolute_error(true_comp, pred_comp)
            overall_metrics['r2'] = r2_score(true_comp.reshape(-1), pred_comp.reshape(-1))
            
            # 2. Calculate metrics for each particle type
            particle_metrics = {}
            for i, col in enumerate(true_comp_cols):
                particle = col.replace('comp_', '')
                true_values = true_comp[:, i]
                pred_values = pred_comp[:, i]
                
                particle_metrics[particle] = {
                    'mse': mean_squared_error(true_values, pred_values),
                    'mae': mean_absolute_error(true_values, pred_values),
                    'r2': r2_score(true_values, pred_values)
                }
                
                # Add statistics about this particle's presence
                particle_metrics[particle]['mean_fraction'] = np.mean(true_values)
                particle_metrics[particle]['max_fraction'] = np.max(true_values)
                particle_metrics[particle]['samples_over_10pct'] = np.sum(true_values > 0.1)
            
            analysis['particle_metrics'] = particle_metrics
            
            # 3. Analyze by preset if provided
            if preset_names is not None and 'preset_name' in df.columns:
                preset_metrics = {}
                
                for preset in preset_names:
                    # Filter data for this preset
                    preset_mask = df['preset_name'] == preset
                    if preset_mask.sum() > 0:
                        preset_true = true_comp[preset_mask]
                        preset_pred = pred_comp[preset_mask]
                        
                        preset_metrics[preset] = {
                            'mse': mean_squared_error(preset_true, preset_pred),
                            'mae': mean_absolute_error(preset_true, preset_pred),
                            'r2': r2_score(preset_true.reshape(-1), preset_pred.reshape(-1)),
                            'sample_count': preset_mask.sum()
                        }
                
                analysis['preset_metrics'] = preset_metrics
            
            # 4. Analyze by experimental condition if available
            if 'condition' in df.columns:
                condition_metrics = {}
                
                for condition in df['condition'].unique():
                    condition_mask = df['condition'] == condition
                    if condition_mask.sum() > 0:
                        condition_true = true_comp[condition_mask]
                        condition_pred = pred_comp[condition_mask]
                        
                        condition_metrics[condition] = {
                            'mse': mean_squared_error(condition_true, condition_pred),
                            'mae': mean_absolute_error(condition_true, condition_pred),
                            'r2': r2_score(condition_true.reshape(-1), condition_pred.reshape(-1)),
                            'sample_count': condition_mask.sum()
                        }
                
                analysis['condition_metrics'] = condition_metrics
        else:
            # Only have predicted compositions, no true values
            # Calculate basic statistics about predictions
            comp_df = pd.DataFrame(pred_comp, columns=[col.replace('comp_', '') for col in true_comp_cols])
            
            particle_stats = {}
            for col in comp_df.columns:
                particle_stats[col] = {
                    'mean_predicted': comp_df[col].mean(),
                    'max_predicted': comp_df[col].max(),
                    'min_predicted': comp_df[col].min(),
                    'std_predicted': comp_df[col].std()
                }
            
            analysis['particle_predictions'] = particle_stats
            
            # If conditions are available, analyze by condition
            if 'condition' in df.columns:
                condition_predictions = {}
                
                for condition in df['condition'].unique():
                    mask = df['condition'] == condition
                    if mask.sum() > 0:
                        condition_pred = pred_comp[mask]
                        condition_df = pd.DataFrame(condition_pred, 
                                                 columns=[col.replace('comp_', '') for col in true_comp_cols])
                        
                        condition_predictions[condition] = {
                            'mean_comp': condition_df.mean().to_dict(),
                            'sample_count': mask.sum()
                        }
                
                analysis['condition_predictions'] = condition_predictions
        
        # Add overall metrics
        analysis['overall'] = overall_metrics
        
        return analysis
        
# Function to create model for beam composition analysis
def create_beam_composition_model(data_generator=None):
    """
    Create a new BeamHaloCompositionModel configured for the data generator
    
    Parameters:
    -----------
    data_generator : BL4SHaloCompositionGenerator
        Generator for beam composition data
        
    Returns:
    --------
    BeamHaloCompositionModel : Configured model
    """
    # Create model with the data generator
    model = BeamHaloCompositionModel(data_generator)
    
    # If data generator is provided, create a small sample dataset to initialize
    if data_generator is not None:
        print("Generating sample data to configure the model...")
        sample_df = data_generator.generate_dataset(n_runs=5, n_events_per_run=50)
        
        # Initialize preprocessing with sample data
        model.preprocess_data(sample_df)
        
        # Build the model architecture
        model.build_model()
        
        print("Model created and configured successfully")
    
    return model

# Function to train and evaluate the composition model
def train_and_evaluate_composition_model(model, data_generator, epochs=50, batch_size=128, save_path=None):
    """
    Train and evaluate a beam composition model
    
    Parameters:
    -----------
    model : BeamHaloCompositionModel
        Model to train
    data_generator : BL4SHaloCompositionGenerator
        Generator for beam composition data
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    save_path : str or None
        Path to save the trained model
        
    Returns:
    --------
    dict : Results including trained model, evaluation metrics and visualizations
    """
    # Generate training data
    print("Generating training data...")
    train_df = data_generator.generate_dataset(n_runs=30, n_events_per_run=500)
    
    # Train the model
    print(f"Training model for {epochs} epochs...")
    history = model.train(
        df=train_df,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Generate testing data with different seed
    print("Generating test data...")
    data_generator.seed = 123  # Different seed for test data
    test_df = data_generator.generate_dataset(n_runs=10, n_events_per_run=300)
    
    # Evaluate the model
    print("Evaluating model...")
    evaluation = model.evaluate(test_df)
    
    # Make predictions
    predictions = model.predict(test_df)
    
    # Create visualizations
    print("Creating visualizations...")
    visualizations = {}
    
    # Training history
    visualizations['training_history'] = model.visualize_training_history()
    
    # Composition predictions
    visualizations['composition_predictions'] = model.visualize_composition_predictions(
        test_df, predictions)
    
    # Halo detector patterns
    visualizations['halo_patterns'] = model.visualize_halo_patterns(test_df)
    
    # Composition correlation
    try:
        visualizations['composition_correlation'] = model.visualize_composition_correlation(test_df)
    except Exception as e:
        print(f"Warning: Could not create composition correlation plot: {e}")
    
    # Evaluate halo contribution
    print("Analyzing halo detector contribution...")
    halo_contribution = model.evaluate_halo_contribution(test_df)
    
    # Halo contribution visualization
    visualizations['halo_contribution'] = model.visualize_halo_contribution(halo_contribution)
    
    # Save model if path is provided
    if save_path:
        print(f"Saving model to {save_path}...")
        model.save(save_path)
    
    # Detailed beam composition analysis
    composition_analysis = model.analyze_beam_composition(test_df, predictions)
    
    # Print summary of results
    print("\nEvaluation Results Summary:")
    print("=" * 50)
    
    if 'composition' in evaluation:
        comp_metrics = evaluation['composition']['overall']
        print(f"Composition Prediction MSE: {comp_metrics['mse']:.4f}")
        print(f"Composition Prediction MAE: {comp_metrics['mae']:.4f}")
        print(f"Composition Prediction R²: {comp_metrics['r2']:.4f}")
    
    if 'average_contribution_pct' in halo_contribution:
        print(f"Halo Detector Contribution: {halo_contribution['average_contribution_pct']:.2f}%")
    
    print("=" * 50)
    
    # Return results
    return {
        'model': model,
        'history': history,
        'evaluation': evaluation,
        'predictions': predictions,
        'halo_contribution': halo_contribution,
        'composition_analysis': composition_analysis,
        'visualizations': visualizations
    }
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, multiply, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from matplotlib.patches import Circle

class BeamHaloCompositionModel:
    """
    Deep learning model for analyzing beam halo composition data from BL4S experiments.
    This model predicts beam composition percentages from halo detector signals.
    """
    
    def __init__(self, data_generator=None):
        """
        Initialize the Beam Halo Composition model
        
        Parameters:
        -----------
        data_generator : BL4SHaloCompositionGenerator
            Generator to create training data
        """
        self.data_generator = data_generator
        self.model = None
        self.encoders = {}
        self.scalers = {}
        self.input_shapes = {}
        self.history = None
        
    def preprocess_data(self, df, for_prediction=False):
        """
        Preprocess raw data from BL4SHaloCompositionGenerator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw data from generator
        for_prediction : bool
            If True, don't expect target variables (for inference)
            
        Returns:
        --------
        tuple : (X, y) processed features and targets
        """
        # Define feature groups based on new data generator format
        halo_features = ['halo_q1_hit', 'halo_q2_hit', 'halo_q3_hit', 'halo_q4_hit', 'any_halo_hit']
        halo_adc_features = ['halo_q1_adc', 'halo_q2_adc', 'halo_q3_adc', 'halo_q4_adc']
        
        # Add tracking features if available
        tracking_features = ['dwc4_x', 'dwc4_y']
        
        # Add experimental condition features
        condition_features = ['scattered_flag', 'magnet_on']
        
        # Handle missing values - replace NaN with 0 for simplicity
        df_clean = df.copy()
        
        # Fill missing values
        for feature_list in [halo_features, halo_adc_features, tracking_features, condition_features]:
            for col in feature_list:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(0)
        
        # Create derived halo features
        halo_derived = self._create_halo_derived_features(df_clean)
        
        # Scale numerical features
        feature_groups = {}
        
        # Check and include available features
        if all(col in df_clean.columns for col in halo_features):
            feature_groups['halo'] = halo_features
        
        if all(col in df_clean.columns for col in halo_adc_features):
            feature_groups['halo_adc'] = halo_adc_features
        
        available_tracking = [col for col in tracking_features if col in df_clean.columns]
        if available_tracking:
            feature_groups['tracking'] = available_tracking
        
        available_condition = [col for col in condition_features if col in df_clean.columns]
        if available_condition:
            feature_groups['condition'] = available_condition
        
        # Create or use scalers for each feature group
        X = {}
        
        for group_name, columns in feature_groups.items():
            if not columns:
                continue
                
            if group_name not in self.scalers or not self.scalers[group_name]:
                # First time - create scalers
                self.scalers[group_name] = StandardScaler()
                X[f'{group_name}_input'] = self.scalers[group_name].fit_transform(df_clean[columns])
            else:
                # Use existing scalers
                X[f'{group_name}_input'] = self.scalers[group_name].transform(df_clean[columns])
        
        # Add halo derived features
        if 'halo_derived' not in self.scalers or not self.scalers['halo_derived']:
            self.scalers['halo_derived'] = StandardScaler()
            X['halo_derived_input'] = self.scalers['halo_derived'].fit_transform(halo_derived)
        else:
            X['halo_derived_input'] = self.scalers['halo_derived'].transform(halo_derived)
        
        # Store input shapes for model creation
        self.input_shapes = {key: val.shape[1:] for key, val in X.items()}
        
        # If prediction mode, just return features
        if for_prediction:
            return X, None
        
        # Prepare targets (particle composition percentages)
        y = {}
        
        # Extract composition target columns
        comp_cols = [col for col in df_clean.columns if col.startswith('comp_')]
        
        if comp_cols:
            y['composition'] = df_clean[comp_cols].values
            
            # Store composition column names for reference
            self.comp_cols = comp_cols
        
        # Additional targets if available
        if 'particle_type' in df_clean.columns:
            if not self.encoders.get('particle_type'):
                self.encoders['particle_type'] = OneHotEncoder(sparse_output=False)
                particle_type_encoded = self.encoders['particle_type'].fit_transform(
                    df_clean['particle_type'].values.reshape(-1, 1)
                )
            else:
                particle_type_encoded = self.encoders['particle_type'].transform(
                    df_clean['particle_type'].values.reshape(-1, 1)
                )
            y['particle_type'] = particle_type_encoded
        
        if 'true_energy' in df_clean.columns:
            y['energy'] = df_clean['true_energy'].values.reshape(-1, 1)
        
        if 'condition' in df_clean.columns:
            if not self.encoders.get('condition'):
                self.encoders['condition'] = OneHotEncoder(sparse_output=False)
                condition_encoded = self.encoders['condition'].fit_transform(
                    df_clean['condition'].values.reshape(-1, 1)
                )
            else:
                condition_encoded = self.encoders['condition'].transform(
                    df_clean['condition'].values.reshape(-1, 1)
                )
            y['condition'] = condition_encoded
        
        return X, y
    
    def _create_halo_derived_features(self, df):
        """
        Create derived features from halo detector data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with halo detector data
            
        Returns:
        --------
        pandas.DataFrame : Derived halo features
        """
        derived = pd.DataFrame(index=df.index)
        
        # Check if halo hit columns are available
        hit_cols = ['halo_q1_hit', 'halo_q2_hit', 'halo_q3_hit', 'halo_q4_hit']
        adc_cols = ['halo_q1_adc', 'halo_q2_adc', 'halo_q3_adc', 'halo_q4_adc']
        
        if all(col in df.columns for col in hit_cols):
            # Extract quadrant hits
            q1 = df['halo_q1_hit'].values
            q2 = df['halo_q2_hit'].values
            q3 = df['halo_q3_hit'].values
            q4 = df['halo_q4_hit'].values
            
            # Horizontal and vertical asymmetry
            derived['h_asymmetry_hit'] = (q1 + q4) - (q2 + q3)
            derived['v_asymmetry_hit'] = (q1 + q2) - (q3 + q4)
            
            # Total quadrant activity
            derived['total_hit_activity'] = q1 + q2 + q3 + q4
            
            # Quadrant patterns
            derived['top_active'] = ((q1 + q2) > 0).astype(float)  # Any top quadrant active
            derived['bottom_active'] = ((q3 + q4) > 0).astype(float)  # Any bottom quadrant active
            derived['left_active'] = ((q2 + q3) > 0).astype(float)  # Any left quadrant active
            derived['right_active'] = ((q1 + q4) > 0).astype(float)  # Any right quadrant active
            
            # Diagonal activation
            derived['diag1_hit'] = q1 + q3  # Top-right + bottom-left
            derived['diag2_hit'] = q2 + q4  # Top-left + bottom-right
        
        # Process ADC values if available
        if all(col in df.columns for col in adc_cols):
            # Extract ADC values
            q1_adc = df['halo_q1_adc'].values
            q2_adc = df['halo_q2_adc'].values
            q3_adc = df['halo_q3_adc'].values
            q4_adc = df['halo_q4_adc'].values
            
            # Total ADC and asymmetries
            derived['total_adc'] = q1_adc + q2_adc + q3_adc + q4_adc
            derived['h_asymmetry_adc'] = (q1_adc + q4_adc) - (q2_adc + q3_adc)
            derived['v_asymmetry_adc'] = (q1_adc + q2_adc) - (q3_adc + q4_adc)
            
            # ADC ratios with smoothing to avoid division by zero
            smooth = 0.01
            derived['h_asymmetry_ratio'] = (derived['h_asymmetry_adc'] + smooth) / (derived['total_adc'] + smooth)
            derived['v_asymmetry_ratio'] = (derived['v_asymmetry_adc'] + smooth) / (derived['total_adc'] + smooth)
            
            # Quadrant ratios
            derived['q1q3_ratio'] = (q1_adc + smooth) / (q3_adc + smooth)
            derived['q2q4_ratio'] = (q2_adc + smooth) / (q4_adc + smooth)
            derived['top_bottom_ratio'] = (q1_adc + q2_adc + smooth) / (q3_adc + q4_adc + smooth)
            derived['left_right_ratio'] = (q2_adc + q3_adc + smooth) / (q1_adc + q4_adc + smooth)
            
            # Diagonal features
            derived['diag1_adc'] = q1_adc + q3_adc  # Top-right + bottom-left
            derived['diag2_adc'] = q2_adc + q4_adc  # Top-left + bottom-right
            derived['diag_ratio'] = (derived['diag1_adc'] + smooth) / (derived['diag2_adc'] + smooth)
        
        return derived

    def _halo_attention_mechanism(self, inputs):
        """
        Attention mechanism that focuses on important halo patterns
        
        Parameters:
        -----------
        inputs : tensor
            Input tensor
            
        Returns:
        --------
        tensor : Output tensor with attention applied
        """
        # First dense layer to create intermediate representation
        x = Dense(32, activation='tanh')(inputs)
        
        # Calculate attention weights
        attention_weights = Dense(inputs.shape[-1], activation='softmax', name='halo_attention_weights')(x)
        
        # Apply attention weights to input
        context_vector = multiply([inputs, attention_weights])
        
        return context_vector, attention_weights
    
    def build_model(self):
        """
        Build the multi-input deep learning model for beam composition prediction
        
        Returns:
        --------
        tensorflow.keras.models.Model : The compiled model
        """
        # Define input layers based on available feature groups
        inputs = {}
        feature_branches = []
        
        # Process each input group if available
        if 'halo_input' in self.input_shapes:
            halo_input = Input(shape=self.input_shapes['halo_input'], name='halo_input')
            inputs['halo_input'] = halo_input
            
            # Apply attention mechanism to halo hits
            x_halo, _ = self._halo_attention_mechanism(halo_input)
            x_halo = Dense(32, activation=LeakyReLU(alpha=0.1))(x_halo)
            feature_branches.append(x_halo)
        
        if 'halo_adc_input' in self.input_shapes:
            halo_adc_input = Input(shape=self.input_shapes['halo_adc_input'], name='halo_adc_input')
            inputs['halo_adc_input'] = halo_adc_input
            
            # Process ADC values
            x_adc = Dense(32, activation=LeakyReLU(alpha=0.1))(halo_adc_input)
            x_adc = BatchNormalization()(x_adc)
            feature_branches.append(x_adc)
        
        if 'halo_derived_input' in self.input_shapes:
            halo_derived_input = Input(shape=self.input_shapes['halo_derived_input'], name='halo_derived_input')
            inputs['halo_derived_input'] = halo_derived_input
            
            # Process derived halo features
            x_derived = Dense(64, activation=LeakyReLU(alpha=0.1))(halo_derived_input)
            feature_branches.append(x_derived)
        
        if 'tracking_input' in self.input_shapes:
            tracking_input = Input(shape=self.input_shapes['tracking_input'], name='tracking_input')
            inputs['tracking_input'] = tracking_input
            
            # Process tracking data
            x_track = Dense(32, activation=LeakyReLU(alpha=0.1))(tracking_input)
            feature_branches.append(x_track)
        
        if 'condition_input' in self.input_shapes:
            condition_input = Input(shape=self.input_shapes['condition_input'], name='condition_input')
            inputs['condition_input'] = condition_input
            
            # Process condition data
            x_cond = Dense(16, activation=LeakyReLU(alpha=0.1))(condition_input)
            feature_branches.append(x_cond)
        
        # Merge all feature branches if we have any
        if feature_branches:
            if len(feature_branches) > 1:
                merged = Concatenate()(feature_branches)
            else:
                merged = feature_branches[0]
            
            # Shared deep layers
            x = Dense(128, activation=LeakyReLU(alpha=0.1))(merged)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            x = Dense(256, activation=LeakyReLU(alpha=0.1))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.4)(x)
            x = Dense(128, activation=LeakyReLU(alpha=0.1))(x)
            x = BatchNormalization()(x)
            
            # Define outputs
            outputs = {}
            losses = {}
            metrics = {}
            loss_weights = {}
            
            # Main task: Beam composition prediction
            if hasattr(self, 'comp_cols'):
                num_particle_types = len(self.comp_cols)
                composition_output = Dense(num_particle_types, activation='softmax', name='composition')(x)
                outputs['composition'] = composition_output
                losses['composition'] = 'mean_squared_error'  # MSE works better for composition fractions
                metrics['composition'] = ['mae', 'mse']  # Both metrics are useful
                loss_weights['composition'] = 1.0
            
            # Add additional outputs if needed
            if 'particle_type' in self.encoders:
                particle_type_branch = Dense(64, activation=LeakyReLU(alpha=0.1))(x)
                num_particle_types = len(self.encoders['particle_type'].categories_[0])
                particle_type_output = Dense(num_particle_types, activation='softmax', name='particle_type')(particle_type_branch)
                outputs['particle_type'] = particle_type_output
                losses['particle_type'] = 'categorical_crossentropy'
                metrics['particle_type'] = ['accuracy']
                loss_weights['particle_type'] = 0.5  # Lower weight for this secondary task
            
            if 'energy' in self.input_shapes:
                energy_branch = Dense(32, activation=LeakyReLU(alpha=0.1))(x)
                energy_output = Dense(1, activation='linear', name='energy')(energy_branch)
                outputs['energy'] = energy_output
                losses['energy'] = 'mean_squared_error'
                metrics['energy'] = ['mae']
                loss_weights['energy'] = 0.3
            
            if 'condition' in self.encoders:
                condition_branch = Dense(32, activation=LeakyReLU(alpha=0.1))(x)
                num_conditions = len(self.encoders['condition'].categories_[0])
                condition_output = Dense(num_conditions, activation='softmax', name='condition')(condition_branch)
                outputs['condition'] = condition_output
                losses['condition'] = 'categorical_crossentropy'
                metrics['condition'] = ['accuracy']
                loss_weights['condition'] = 0.4
            
            # Create model
            model = Model(
                inputs=list(inputs.values()),
                outputs=list(outputs.values())
            )
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=losses,
                metrics=metrics,
                loss_weights=loss_weights
            )
            
            self.model = model
            return model
        else:
            raise ValueError("No valid input features found to build the model")
    
    def train(self, df=None, test_size=0.2, epochs=100, batch_size=64, verbose=1, use_class_weights=False):
        """
        Train the model using data from generator or provided dataframe
        
        Parameters:
        -----------
        df : pandas.DataFrame or None
            If provided, use this data instead of generating new data
        test_size : float
            Fraction of data to use for validation
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : int
            Verbosity level (0, 1, or 2)
        use_class_weights : bool
            Whether to use class weights for imbalanced classification tasks
            
        Returns:
        --------
        history : Training history
        """
        # Generate or use provided data
        if df is None and self.data_generator is not None:
            print("Generating synthetic training data...")
            # Use the BL4SHaloCompositionGenerator to create data
            df = self.data_generator.generate_dataset(n_runs=30, n_events_per_run=500)
        
        if df is None:
            raise ValueError("Either provide a dataframe or a data generator")
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Split data into train and validation sets
        X_train = {}
        X_val = {}
        y_train = {}
        y_val = {}
        
        # Get indices for split
        indices = np.arange(len(df))
        train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=42)
        
        # Split features
        for key in X:
            X_train[key] = X[key][train_indices]
            X_val[key] = X[key][val_indices]
        
        # Split targets
        for key in y:
            if isinstance(y[key], np.ndarray):
                y_train[key] = y[key][train_indices]
                y_val[key] = y[key][val_indices]
        
        # Prepare class weights for classification tasks if needed
        class_weights = {}
        if use_class_weights:
            # We're mainly dealing with regression for composition, so this is mostly
            # for any auxiliary classification tasks
            if 'particle_type' in y and 'particle_type' in self.encoders:
                particle_type_indices = np.argmax(y['particle_type'], axis=1)
                particle_type_weights = compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(particle_type_indices),
                    y=particle_type_indices[train_indices]
                )
                particle_type_class_weights = {i: weight for i, weight in enumerate(particle_type_weights)}
                class_weights['particle_type'] = particle_type_class_weights
            
            if 'condition' in y and 'condition' in self.encoders:
                condition_indices = np.argmax(y['condition'], axis=1)
                condition_weights = compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(condition_indices),
                    y=condition_indices[train_indices]
                )
                condition_class_weights = {i: weight for i, weight in enumerate(condition_weights)}
                class_weights['condition'] = condition_class_weights
        
        # Create output directory for model checkpoints
        os.makedirs("model_checkpoints", exist_ok=True)
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath='model_checkpoints/composition_model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(log_dir='./logs/beam_halo_composition')
        ]
        
        # Train the model
        if use_class_weights and class_weights:
            print("Using class weights for imbalanced classification tasks.")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose,
                class_weight=class_weights
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        
        self.history = history
        return history
    
    def predict(self, df):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Data to predict on
            
        Returns:
        --------
        dict : Predictions for each task
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess data
        X, _ = self.preprocess_data(df, for_prediction=True)
        
        # Make predictions
        raw_predictions = self.model.predict(X)
        
        # Format predictions
        results = {}
        
        # Convert output list to dictionary based on model output names
        output_names = [output.name for output in self.model.outputs]
        
        if isinstance(raw_predictions, list):
            for i, name in enumerate(output_names):
                results[name] = raw_predictions[i]
        else:
            # Single output case
            results[output_names[0]] = raw_predictions
        
        # Add composition columns if we have them
        if 'composition' in results and hasattr(self, 'comp_cols'):
            # Format composition results with column names
            composition_df = pd.DataFrame(
                results['composition'], 
                columns=[col.replace('comp_', '') for col in self.comp_cols]
            )
            results['composition_df'] = composition_df
        
        # Convert categorical outputs back to labels where applicable
        if 'particle_type' in results and 'particle_type' in self.encoders:
            particle_type_indices = np.argmax(results['particle_type'], axis=1)
            results['particle_type_labels'] = self.encoders['particle_type'].categories_[0][particle_type_indices]
        
        if 'condition' in results and 'condition' in self.encoders:
            condition_indices = np.argmax(results['condition'], axis=1)
            results['condition_labels'] = self.encoders['condition'].categories_[0][condition_indices]
        
        return results
    
    def evaluate(self, df):
        """
        Evaluate model performance on test data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Test data
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Evaluate model
        results = self.model.evaluate(X, y, verbose=1)
        
        # Make predictions for detailed metrics
        predictions = self.predict(df)
        
        # Calculate detailed metrics
        metrics = {}
        
        # Overall model metrics (from evaluate)
        metrics['model'] = dict(zip(self.model.metrics_names, results))
        
        # Composition prediction metrics
        if 'composition' in y and 'composition' in predictions:
            true_comp = y['composition']
            pred_comp = predictions['composition']
            
            # Calculate MSE, MAE, and R² for each particle type
            composition_metrics = {}
            
            for i, col in enumerate(self.comp_cols):
                particle = col.replace('comp_', '')
                composition_metrics[particle] = {
                    'mse': mean_squared_error(true_comp[:, i], pred_comp[:, i]),
                    'mae': mean_absolute_error(true_comp[:, i], pred_comp[:, i]),
                    'r2': r2_score(true_comp[:, i], pred_comp[:, i])
                }
            
            # Overall composition metrics
            composition_metrics['overall'] = {
                'mse': mean_squared_error(true_comp, pred_comp),
                'mae': mean_absolute_error(true_comp, pred_comp),
                'r2': r2_score(true_comp.reshape(-1), pred_comp.reshape(-1))
            }
            
            metrics['composition'] = composition_metrics
        
        # Add metrics for other tasks if available
        if 'particle_type' in y and 'particle_type_labels' in predictions:
            true_particle_types = df['particle_type'].values
            pred_particle_types = predictions['particle_type_labels']
            
            metrics['particle_type'] = {
                'accuracy': np.mean(true_particle_types == pred_particle_types),
                'confusion_matrix': confusion_matrix(true_particle_types, pred_particle_types),
                'classification_report': classification_report(true_particle_types, pred_particle_types, output_dict=True)
            }
        
        if 'energy' in y and 'energy' in predictions:
            true_energy = df['true_energy'].values
            pred_energy = predictions['energy'].flatten()
            
            metrics['energy'] = {
                'mse': mean_squared_error(true_energy, pred_energy),
                'mae': mean_absolute_error(true_energy, pred_energy),
                'r2': r2_score(true_energy, pred_energy)
            }
        
        if 'condition' in y and 'condition_labels' in predictions:
            true_conditions = df['condition'].values
            pred_conditions = predictions['condition_labels']
            
            metrics['condition'] = {
                'accuracy': np.mean(true_conditions == pred_conditions),
                'confusion_matrix': confusion_matrix(true_conditions, pred_conditions),
                'classification_report': classification_report(true_conditions, pred_conditions, output_dict=True)
            }
            
        return metrics
    
    def evaluate_halo_contribution(self, df):
        """
        Evaluate how much the halo detector contributes to composition predictions
        by comparing full model to models with masked halo inputs
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Test data
            
        Returns:
        --------
        dict : Halo detector contribution analysis including MSE and R² metrics
               with and without halo detector data, and percentage contributions
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Regular predictions with full model
        full_predictions = self.predict(df)
        
        # Create a copy of X with masked halo inputs
        X_no_halo = X.copy()
        
        # Identify and mask halo-related inputs
        halo_inputs = ['halo_input', 'halo_adc_input', 'halo_derived_input']
        for key in halo_inputs:
            if key in X_no_halo:
                X_no_halo[key] = np.zeros_like(X[key])
        
        # Make predictions without halo information
        no_halo_raw_predictions = self.model.predict(X_no_halo)
        
        # Format no-halo predictions
        no_halo_predictions = {}
        output_names = [output.name for output in self.model.outputs]
        
        if isinstance(no_halo_raw_predictions, list):
            for i, name in enumerate(output_names):
                no_halo_predictions[name] = no_halo_raw_predictions[i]
        else:
            # Single output case
            no_halo_predictions[output_names[0]] = no_halo_raw_predictions
        
        # Calculate contribution metrics
        contribution = {}
        
        # 1. Impact on composition prediction
        if 'composition' in full_predictions and 'composition' in no_halo_predictions and 'composition' in y:
            full_comp = full_predictions['composition']
            no_halo_comp = no_halo_predictions['composition']
            true_comp = y['composition']
            
            # Calculate metrics with and without halo for each particle type
            particle_contributions = {}
            
            for i, col in enumerate(self.comp_cols):
                particle = col.replace('comp_', '')
                
                # MSE with and without halo
                full_mse = mean_squared_error(true_comp[:, i], full_comp[:, i])
                no_halo_mse = mean_squared_error(true_comp[:, i], no_halo_comp[:, i])
                
                # R² with and without halo
                full_r2 = r2_score(true_comp[:, i], full_comp[:, i])
                no_halo_r2 = r2_score(true_comp[:, i], no_halo_comp[:, i])
                
                # Calculate contribution to accuracy
                # For MSE (lower is better), calculate how much error increases without halo
                mse_increase = max(0, no_halo_mse - full_mse)
                mse_contribution = mse_increase / max(0.0001, no_halo_mse) * 100
                
                # For R² (higher is better), calculate how much R² decreases without halo
                r2_decrease = max(0, full_r2 - no_halo_r2)
                r2_contribution = r2_decrease / max(0.0001, full_r2) * 100
                
                particle_contributions[particle] = {
                    'full_mse': full_mse,
                    'no_halo_mse': no_halo_mse,
                    'mse_contribution_pct': mse_contribution,
                    'full_r2': full_r2,
                    'no_halo_r2': no_halo_r2,
                    'r2_contribution_pct': r2_contribution
                }
            
            # Overall composition metrics
            overall_full_mse = mean_squared_error(true_comp, full_comp)
            overall_no_halo_mse = mean_squared_error(true_comp, no_halo_comp)
            overall_full_r2 = r2_score(true_comp.reshape(-1), full_comp.reshape(-1))
            overall_no_halo_r2 = r2_score(true_comp.reshape(-1), no_halo_comp.reshape(-1))
            
            # Calculate overall contribution
            overall_mse_increase = max(0, overall_no_halo_mse - overall_full_mse)
            overall_mse_contribution = overall_mse_increase / max(0.0001, overall_no_halo_mse) * 100
            
            overall_r2_decrease = max(0, overall_full_r2 - overall_no_halo_r2)
            overall_r2_contribution = overall_r2_decrease / max(0.0001, overall_full_r2) * 100
            
            particle_contributions['overall'] = {
                'full_mse': overall_full_mse,
                'no_halo_mse': overall_no_halo_mse,
                'mse_contribution_pct': overall_mse_contribution,
                'full_r2': overall_full_r2,
                'no_halo_r2': overall_no_halo_r2,
                'r2_contribution_pct': overall_r2_contribution
            }
            
            contribution['composition'] = particle_contributions
        
        # 2. Impact on any secondary tasks (if available)
        if 'particle_type' in full_predictions and 'particle_type' in no_halo_predictions and 'particle_type_labels' in full_predictions:
            full_particle_type = np.argmax(full_predictions['particle_type'], axis=1)
            no_halo_particle_type = np.argmax(no_halo_predictions['particle_type'], axis=1)
            
            if 'particle_type' in y:
                true_particle_type = np.argmax(y['particle_type'], axis=1)
                
                # Calculate accuracy with and without halo
                full_accuracy = np.mean(full_particle_type == true_particle_type)
                no_halo_accuracy = np.mean(no_halo_particle_type == true_particle_type)
                
                # Calculate halo contribution to particle classification
                accuracy_decrease = max(0, full_accuracy - no_halo_accuracy)
                particle_type_contribution = accuracy_decrease / max(0.0001, full_accuracy) * 100
                
                contribution['particle_type'] = {
                    'full_accuracy': full_accuracy,
                    'no_halo_accuracy': no_halo_accuracy,
                    'contribution_pct': particle_type_contribution
                }
        
        # 3. Impact on energy prediction (if available)
        if 'energy' in full_predictions and 'energy' in no_halo_predictions and 'energy' in y:
            full_energy = full_predictions['energy'].flatten()
            no_halo_energy = no_halo_predictions['energy'].flatten()
            true_energy = y['energy'].flatten()
            
            # Calculate MSE with and without halo
            full_mse = mean_squared_error(true_energy, full_energy)
            no_halo_mse = mean_squared_error(true_energy, no_halo_energy)
            
            # Calculate R² with and without halo
            full_r2 = r2_score(true_energy, full_energy)
            no_halo_r2 = r2_score(true_energy, no_halo_energy)
            
            # Calculate contribution
            mse_increase = max(0, no_halo_mse - full_mse)
            mse_contribution = mse_increase / max(0.0001, no_halo_mse) * 100
            
            r2_decrease = max(0, full_r2 - no_halo_r2)
            r2_contribution = r2_decrease / max(0.0001, full_r2) * 100
            
            contribution['energy'] = {
                'full_mse': full_mse,
                'no_halo_mse': no_halo_mse,
                'mse_contribution_pct': mse_contribution,
                'full_r2': full_r2,
                'no_halo_r2': no_halo_r2,
                'r2_contribution_pct': r2_contribution
            }
        
        # Calculate average contribution across all tasks
        if contribution:
            # Extract contribution percentages from each task
            r2_contributions = []
            
            if 'composition' in contribution:
                r2_contributions.append(contribution['composition']['overall']['r2_contribution_pct'])
            
            if 'particle_type' in contribution:
                r2_contributions.append(contribution['particle_type']['contribution_pct'])
            
            if 'energy' in contribution:
                r2_contributions.append(contribution['energy']['r2_contribution_pct'])
            
            if r2_contributions:
                contribution['average_contribution_pct'] = np.mean(r2_contributions)
        
        return contribution