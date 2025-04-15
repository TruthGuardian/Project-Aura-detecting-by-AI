import numpy as np
import pandas as pd
from scipy.stats import norm, expon
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from IPython.display import HTML

class BL4SHaloCompositionGenerator:
    """
    Data generator for the BL4S Beam Halo experiment simulation with 
    explicit tracking of beam composition percentages.
    
    This improved generator creates data suitable for training models to predict 
    beam composition percentages from halo detector signals.
    """
    
    def __init__(self, seed=42):
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Detector positions (z-axis in meters)
        self.z_positions = {
            's1': 0.0,           # First scintillator (for trigger)
            'dwc1': 0.5,         # First Delay Wire Chamber
            'c1': 1.0,           # First Cherenkov detector
            'dwc2': 1.5,         # Second Delay Wire Chamber
            'target': 2.0,       # Interaction region or target position
            'dipole_magnet': 2.5, # Dipole magnet for particle deflection
            'halo_detector': 3.0, # Halo detector (4-quadrant)
            'dwc3': 3.5,         # Third Delay Wire Chamber
            'c2': 4.0,           # Second Cherenkov detector
            'dwc4': 4.5,         # Fourth Delay Wire Chamber
            's2': 5.0,           # Second scintillator (for ToF)
            'calorimeter': 5.5   # Calorimeter for energy measurement
        }
        
        # Beam parameters (defaults)
        self.beam_params = {
            'energy': 5.0,  # GeV
            'beam_center_x': 0.0,  # mm
            'beam_center_y': 0.0,  # mm
            'beam_sigma_x': 2.0,  # mm
            'beam_sigma_y': 2.0,  # mm
            'beam_div_x': 0.5,  # mrad
            'beam_div_y': 0.5,  # mrad
        }
        
        # Detector parameters
        self.detector_params = {
            'dwc_resolution': 0.2,  # mm
            'dwc_efficiency': 0.98,  # Detection efficiency
            'halo_hole_radius': 10.0,  # mm, central hole in halo detector
            'halo_outer_radius': 50.0,  # mm, outer radius of halo detector
            'halo_adc_noise': 20.0,  # ADC noise level
            'halo_adc_gain': 10.0,  # ADC gain per hit
            'c1_threshold': 2.0,  # GeV, threshold for Cherenkov 1
            'c2_threshold': 3.5,  # GeV, threshold for Cherenkov 2
            'tof_resolution': 0.1,  # ns, time resolution
            'calo_resolution': 0.1,  # Fractional energy resolution
            'scintillator_efficiency': 0.99,  # Detection efficiency for scintillators
            'magnet_field_strength': 0.8,  # Tesla, dipole magnet field strength
        }
        
        # Define particle types and their properties
        self.particle_properties = {
            'proton': {
                'mass': 0.938,    # GeV/c²
                'charge': 1.0,    # e
                'halo_response': 1.0,  # Relative response in halo detector
                'color': 'blue',
                'symbol': 'p'
            },
            'electron': {
                'mass': 0.000511,
                'charge': -1.0,
                'halo_response': 0.6,
                'color': 'red',
                'symbol': 'e-'
            },
            'pion+': {
                'mass': 0.140,
                'charge': 1.0,
                'halo_response': 0.9,
                'color': 'green',
                'symbol': 'π+'
            },
            'pion-': {
                'mass': 0.140,
                'charge': -1.0,
                'halo_response': 0.9,
                'color': 'limegreen',
                'symbol': 'π-'
            },
            'kaon+': {
                'mass': 0.494,
                'charge': 1.0,
                'halo_response': 1.2,
                'color': 'purple',
                'symbol': 'K+'
            },
            'kaon-': {
                'mass': 0.494,
                'charge': -1.0,
                'halo_response': 1.2,
                'color': 'violet',
                'symbol': 'K-'
            },
            'muon+': {
                'mass': 0.106,
                'charge': 1.0,
                'halo_response': 0.7,
                'color': 'orange',
                'symbol': 'μ+'
            },
            'muon-': {
                'mass': 0.106,
                'charge': -1.0,
                'halo_response': 0.7,
                'color': 'darkorange',
                'symbol': 'μ-'
            }
        }
        
        # Create beam composition presets that will be used to generate mixed beams
        self.beam_compositions = {
            # Pure beams (for testing)
            'Pure_Proton': {
                'proton': 1.0,
                'electron': 0.0,
                'pion+': 0.0,
                'pion-': 0.0,
                'kaon+': 0.0,
                'kaon-': 0.0,
                'muon+': 0.0,
                'muon-': 0.0
            },
            'Pure_Electron': {
                'proton': 0.0,
                'electron': 1.0,
                'pion+': 0.0,
                'pion-': 0.0,
                'kaon+': 0.0,
                'kaon-': 0.0,
                'muon+': 0.0,
                'muon-': 0.0
            },
            # Mixed beams with different compositions
            'Proton_Rich': {
                'proton': 0.7,
                'electron': 0.05,
                'pion+': 0.1,
                'pion-': 0.05,
                'kaon+': 0.05,
                'kaon-': 0.0,
                'muon+': 0.03,
                'muon-': 0.02
            },
            'Lepton_Rich': {
                'proton': 0.2,
                'electron': 0.4,
                'pion+': 0.1,
                'pion-': 0.05,
                'kaon+': 0.05,
                'kaon-': 0.05,
                'muon+': 0.1,
                'muon-': 0.05
            },
            'Balanced_Beam': {
                'proton': 0.25,
                'electron': 0.15,
                'pion+': 0.15,
                'pion-': 0.15,
                'kaon+': 0.1,
                'kaon-': 0.1,
                'muon+': 0.05,
                'muon-': 0.05
            },
            'Kaon_Enhanced': {
                'proton': 0.3,
                'electron': 0.1,
                'pion+': 0.15,
                'pion-': 0.05,
                'kaon+': 0.2,
                'kaon-': 0.1,
                'muon+': 0.05,
                'muon-': 0.05
            },
            'Muon_Enhanced': {
                'proton': 0.3,
                'electron': 0.1,
                'pion+': 0.1,
                'pion-': 0.1,
                'kaon+': 0.05,
                'kaon-': 0.05,
                'muon+': 0.2,
                'muon-': 0.1
            },
            'Pion_Dominated': {
                'proton': 0.2,
                'electron': 0.05,
                'pion+': 0.35,
                'pion-': 0.25,
                'kaon+': 0.05,
                'kaon-': 0.03,
                'muon+': 0.04,
                'muon-': 0.03
            }
        }
        
        # Define experimental conditions
        self.conditions = {
            'Baseline_Nominal': {
                'description': 'Nominal beam with no target',
                'target_material': None,
                'magnet_on': False,
                'scattering_prob': 0.05,  # Background scattering
                'scattering_sigma': 1.0,  # mrad
                'beam_div_factor': 1.0,  # No change to beam divergence
            },
            'Target_C_Foil': {
                'description': 'Carbon foil target',
                'target_material': 'Carbon',
                'magnet_on': False,
                'scattering_prob': 0.3,
                'scattering_sigma': 3.0,  # mrad
                'beam_div_factor': 1.0,
            },
            'Target_Al_Foil': {
                'description': 'Aluminum foil target',
                'target_material': 'Aluminum',
                'magnet_on': False,
                'scattering_prob': 0.4,
                'scattering_sigma': 5.0,  # mrad
                'beam_div_factor': 1.0,
            },
            'Target_Cu_Foil': {
                'description': 'Copper foil target',
                'target_material': 'Copper',
                'magnet_on': False,
                'scattering_prob': 0.5,
                'scattering_sigma': 7.0,  # mrad
                'beam_div_factor': 1.0,
            },
            'Target_C_Foil_Magnet': {
                'description': 'Carbon foil target with magnet on',
                'target_material': 'Carbon',
                'magnet_on': True,
                'scattering_prob': 0.3,
                'scattering_sigma': 3.0,  # mrad
                'beam_div_factor': 1.0,
            },
            'Target_Al_Foil_Magnet': {
                'description': 'Aluminum foil target with magnet on',
                'target_material': 'Aluminum',
                'magnet_on': True,
                'scattering_prob': 0.4,
                'scattering_sigma': 5.0,  # mrad
                'beam_div_factor': 1.0,
            },
            'Target_Cu_Foil_Magnet': {
                'description': 'Copper foil target with magnet on',
                'target_material': 'Copper',
                'magnet_on': True,
                'scattering_prob': 0.5,
                'scattering_sigma': 7.0,  # mrad
                'beam_div_factor': 1.0,
            },
            'Beam_Halo_Study': {
                'description': 'Special setup for beam halo study',
                'target_material': None,
                'magnet_on': True,
                'scattering_prob': 0.15,
                'scattering_sigma': 2.0,  # mrad
                'beam_div_factor': 2.0,  # Deliberately increased divergence
            }
        }
    
    def generate_random_composition(self):
        """
        Generate a random beam composition with valid proportions that sum to 1.0
        """
        # Generate random values for each particle type
        particles = list(self.particle_properties.keys())
        random_values = np.random.rand(len(particles))
        
        # Normalize to sum to 1.0
        composition = random_values / random_values.sum()
        
        # Create a dictionary
        comp_dict = {particle: comp for particle, comp in zip(particles, composition)}
        
        return comp_dict
    
    def create_custom_composition(self, preset_name=None, modify_preset=False, custom_dict=None):
        """
        Create a beam composition based on presets, with optional modifications
        
        Parameters:
        - preset_name: Name of a preset composition to use as a base
        - modify_preset: If True, adds random variations to the preset
        - custom_dict: Custom composition dictionary (overrides preset)
        
        Returns:
        - A dictionary mapping particle types to composition fractions
        """
        if custom_dict is not None:
            # Use provided custom composition
            composition = custom_dict.copy()
        elif preset_name in self.beam_compositions:
            # Use a preset composition
            composition = self.beam_compositions[preset_name].copy()
            
            # Add slight random variations if requested
            if modify_preset:
                particles = list(composition.keys())
                # Add random noise to each value
                for particle in particles:
                    composition[particle] += np.random.uniform(-0.05, 0.05)
                    composition[particle] = max(0, composition[particle])  # Ensure non-negative
                
                # Normalize to sum to 1.0
                total = sum(composition.values())
                for particle in particles:
                    composition[particle] /= total
        else:
            # Generate a completely random composition
            composition = self.generate_random_composition()
            
        return composition
    
    def generate_initial_particle(self, condition, energy=5.0, particle_type=None):
        """
        Generate initial particle properties at the source
        
        Parameters:
        - condition: Experimental condition name
        - energy: Beam energy in GeV
        - particle_type: Type of particle to generate (random if None)
        
        Returns:
        - Dictionary containing particle properties
        """
        # Get condition-specific parameters
        condition_params = self.conditions[condition]
        
        # Apply condition-specific beam divergence factor
        div_x = self.beam_params['beam_div_x'] * condition_params['beam_div_factor']
        div_y = self.beam_params['beam_div_y'] * condition_params['beam_div_factor']
        
        # Sample initial position from 2D Gaussian
        x0 = np.random.normal(self.beam_params['beam_center_x'], self.beam_params['beam_sigma_x'])
        y0 = np.random.normal(self.beam_params['beam_center_y'], self.beam_params['beam_sigma_y'])
        
        # Sample initial angles from 2D Gaussian (in mrad)
        theta_x = np.random.normal(0, div_x)
        theta_y = np.random.normal(0, div_y)
        
        # Energy with appropriate spread
        energy_spread = 0.05 * energy
        energy = np.random.normal(energy, energy_spread)
        
        # Determine particle type if not specified
        if particle_type is None:
            particle_types = list(self.particle_properties.keys())
            # Equal probabilities for each type
            weights = [1/len(particle_types)] * len(particle_types)
            particle_type = np.random.choice(particle_types, p=weights)
        
        # Get particle properties
        properties = self.particle_properties[particle_type]
        
        # Calculate momentum from energy and mass
        mass = properties['mass']
        momentum = np.sqrt(energy**2 - mass**2)
        
        return {
            'x0': x0,
            'y0': y0,
            'theta_x': theta_x,
            'theta_y': theta_y,
            'energy': energy,
            'particle_type': particle_type,
            'scattered': False,
            'momentum': momentum,
            'charge': properties['charge'],
            'halo_response': properties['halo_response']
        }
    
    def propagate_to_z(self, particle, z_target, consider_magnet=True):
        """
        Propagate particle from z=0 to target z position with optional magnetic field effects
        """
        # Extract particle properties
        x0, y0 = particle['x0'], particle['y0']
        theta_x, theta_y = particle['theta_x'], particle['theta_y']
        
        # Convert angles from mrad to rad for calculation
        theta_x_rad = theta_x / 1000.0
        theta_y_rad = theta_y / 1000.0
        
        # Check if we need to consider magnetic field effects
        if consider_magnet and particle.get('passed_magnet', False):
            # Particle has passed through a magnetic field, use deflected angles
            theta_x_rad = particle.get('deflected_theta_x', theta_x_rad)
            theta_y_rad = particle.get('deflected_theta_y', theta_y_rad)
        
        # Simple linear propagation
        x = x0 + z_target * np.tan(theta_x_rad) * 1000  # Convert back to mm
        y = y0 + z_target * np.tan(theta_y_rad) * 1000  # Convert back to mm
        
        return x, y
    
    def apply_magnetic_field(self, particle, condition):
        """
        Apply magnetic field effects to particle trajectory
        """
        # Check if magnet is on for this condition
        condition_params = self.conditions[condition]
        if not condition_params['magnet_on']:
            # No magnetic field effects
            particle['passed_magnet'] = True
            return particle
        
        # Get magnet parameters
        B = self.detector_params['magnet_field_strength']  # Tesla
        
        # Get particle properties
        p = particle['momentum']  # GeV/c
        q = particle['charge']    # e (elementary charge)
        
        # Calculate deflection (in rad)
        # Simplified calculation: deflection angle = qBL/p
        # We'll assume magnet length L = 0.3 meters
        L = 0.3  # meters
        
        # Convert units: B(T) * L(m) * q(e) / p(GeV/c) * 0.3 to get angle in rad
        # constant factor ~0.3 converts T·m·e/GeV/c to rad
        deflection_angle = q * B * L * 0.3 / p
        
        # In our setup, let's say the magnetic field is in the y-direction
        # So it deflects particles in the x-direction
        # original theta_x is in mrad, convert to rad for calculation
        theta_x_rad = particle['theta_x'] / 1000.0
        
        # Apply deflection
        particle['deflected_theta_x'] = theta_x_rad + deflection_angle
        particle['deflected_theta_y'] = particle['theta_y'] / 1000.0  # No deflection in y
        particle['passed_magnet'] = True
        
        return particle
    
    def simulate_interaction(self, particle, condition):
        """
        Simulate particle interaction with target or collimator
        """
        # Get condition-specific parameters
        condition_params = self.conditions[condition]
        
        # Check if particle scatters based on probability
        if np.random.random() < condition_params['scattering_prob']:
            # Particle is scattered
            particle['scattered'] = True
            
            # Sample scattering angles (additional angles to add)
            scatter_angle_x = np.random.normal(0, condition_params['scattering_sigma'])
            scatter_angle_y = np.random.normal(0, condition_params['scattering_sigma'])
            
            # Update particle angles (mrad)
            particle['theta_x'] += scatter_angle_x
            particle['theta_y'] += scatter_angle_y
            
            # Small energy loss (more for heavier materials)
            if condition_params['target_material'] == 'Carbon':
                energy_loss_factor = 0.02
            elif condition_params['target_material'] == 'Aluminum':
                energy_loss_factor = 0.04
            elif condition_params['target_material'] == 'Copper':
                energy_loss_factor = 0.06
            else:  # No material
                energy_loss_factor = 0.005
                
            # Apply energy loss
            energy_loss = np.random.exponential(energy_loss_factor * particle['energy'])
            particle['energy'] -= min(energy_loss, 0.5 * particle['energy'])  # Limit max energy loss
            
            # Update momentum after energy loss
            mass = self.particle_properties[particle['particle_type']]['mass']
            particle['momentum'] = np.sqrt(max(0, particle['energy']**2 - mass**2))
        
        return particle
    
    def simulate_halo_detector(self, x, y, particle):
        """
        Simulate 4-quadrant halo detector response with ADC values
        """
        # Calculate radius from beam center
        r = np.sqrt(x**2 + y**2)
        
        # Initialize quadrant hit flags and ADC values
        # Quadrants: top-right, top-left, bottom-left, bottom-right
        quadrant_hits = [0, 0, 0, 0]
        quadrant_adc = [0, 0, 0, 0]
        
        # If particle is within detector sensitive area (annular region)
        if r > self.detector_params['halo_hole_radius'] and r < self.detector_params['halo_outer_radius']:
            # Determine quadrant
            if x >= 0 and y >= 0:  # Top-right
                q_idx = 0
            elif x < 0 and y >= 0:  # Top-left
                q_idx = 1
            elif x < 0 and y < 0:  # Bottom-left
                q_idx = 2
            else:  # Bottom-right
                q_idx = 3
                
            # Register hit
            quadrant_hits[q_idx] = 1
            
            # Calculate ADC value based on particle properties and position
            # Particles closer to the outer edge tend to deposit more energy
            r_normalized = (r - self.detector_params['halo_hole_radius']) / (
                self.detector_params['halo_outer_radius'] - self.detector_params['halo_hole_radius'])
            
            # More energy deposition near the edges
            edge_factor = 1.0 + 0.5 * r_normalized
            
            # Scale by particle's halo response factor
            response_factor = particle['halo_response']
            
            # Base ADC value
            base_adc = 100 + 300 * r_normalized
            
            # Add energy dependence
            energy_factor = 0.8 + 0.4 * (particle['energy'] / 5.0)
            
            # Calculate final ADC value with some randomness
            adc_value = base_adc * edge_factor * response_factor * energy_factor
            adc_value = np.random.normal(adc_value, self.detector_params['halo_adc_noise'])
            adc_value = max(0, adc_value)  # Ensure non-negative
            
            # Set the ADC value for the hit quadrant
            quadrant_adc[q_idx] = adc_value
            
            # Add some cross-talk to adjacent quadrants
            for adj_q in range(4):
                if adj_q != q_idx:
                    # Distance between quadrants (0=adjacent, 2=opposite)
                    q_distance = min((adj_q - q_idx) % 4, (q_idx - adj_q) % 4)
                    if q_distance == 1:  # Adjacent quadrants
                        crosstalk = np.random.uniform(0.05, 0.15) * adc_value
                        quadrant_adc[adj_q] = crosstalk
        else:
            # Add some noise to all quadrants
            for q in range(4):
                quadrant_adc[q] = max(0, np.random.normal(0, self.detector_params['halo_adc_noise']))
        
        return quadrant_hits, quadrant_adc
    
    def simulate_event(self, run_id, event_id, condition, composition, energy=5.0):
        """
        Simulate a complete event through the experimental setup
        
        Parameters:
        - run_id: Identifier for the experimental run
        - event_id: Identifier for the event within the run
        - condition: Experimental condition name
        - composition: Dictionary of particle type compositions
        - energy: Beam energy in GeV
        
        Returns:
        - Dictionary containing event data
        """
        # Sample particle type from the given composition
        particle_types = list(composition.keys())
        particle_weights = [composition[p] for p in particle_types]
        particle_type = np.random.choice(particle_types, p=particle_weights)
        
        # Generate initial particle
        particle = self.generate_initial_particle(condition, energy, particle_type)
        
        # Simulate interaction with target/material
        particle = self.simulate_interaction(particle, condition)
        
        # Apply magnetic field effects
        particle = self.apply_magnetic_field(particle, condition)
        
        # Propagate to halo detector
        x_halo, y_halo = self.propagate_to_z(particle, self.z_positions['halo_detector'], consider_magnet=True)
        halo_hits, halo_adc = self.simulate_halo_detector(x_halo, y_halo, particle)
        
        # Calculate track parameters
        # Final position at downstream tracking plane
        x_dwc4, y_dwc4 = self.propagate_to_z(particle, self.z_positions['dwc4'], consider_magnet=True)
        
        # Create event dictionary
        event = {
            'run_id': run_id,
            'event_id': event_id,
            'condition': condition,
            'particle_type': particle_type,
            
            # Halo detector signals
            'halo_q1_hit': halo_hits[0],
            'halo_q2_hit': halo_hits[1],
            'halo_q3_hit': halo_hits[2],
            'halo_q4_hit': halo_hits[3],
            'halo_q1_adc': halo_adc[0],
            'halo_q2_adc': halo_adc[1],
            'halo_q3_adc': halo_adc[2],
            'halo_q4_adc': halo_adc[3],
            'any_halo_hit': int(sum(halo_hits) > 0),
            
            # Tracking info
            'dwc4_x': x_dwc4,
            'dwc4_y': y_dwc4,
            'scattered_flag': int(particle['scattered']),
            'magnet_on': int(self.conditions[condition]['magnet_on']),
            
            # True parameters
            'true_energy': particle['energy'],
            'true_momentum': particle['momentum'],
            'true_theta_x': particle['theta_x'],
            'true_theta_y': particle['theta_y'],
            
            # Important: Add the true beam composition percentages as features
            # These act as the target variables for ML models predicting composition
        }
        
        # Add composition percentages to the event
        for p_type, p_fraction in composition.items():
            event[f'comp_{p_type}'] = p_fraction
        
        return event
    
    def generate_dataset(self, n_runs=20, n_events_per_run=1000):
        """
        Generate a dataset with multiple runs, each with a different beam composition
        
        Parameters:
        - n_runs: Number of different experimental runs
        - n_events_per_run: Number of events per run
        
        Returns:
        - DataFrame containing all events with true composition percentages
        """
        all_events = []
        conditions = list(self.conditions.keys())
        
        print(f"Generating {n_runs} runs with {n_events_per_run} events each...")
        
        for run_id in range(n_runs):
            # Determine condition for this run
            condition = np.random.choice(conditions)
            
            # For some runs, use preset compositions, for others use random ones
            use_preset = np.random.random() < 0.7  # 70% chance to use a preset
            
            if use_preset:
                preset_names = list(self.beam_compositions.keys())
                preset_name = np.random.choice(preset_names)
                # Sometimes add small random variations to presets
                modify_preset = np.random.random() < 0.5
                composition = self.create_custom_composition(preset_name, modify_preset)
                composition_name = f"{preset_name}" + ("_modified" if modify_preset else "")
            else:
                composition = self.generate_random_composition()
                composition_name = "Random_Composition"
            
            print(f"Run {run_id}: {condition}, {composition_name}")
            print(f"  Composition: {', '.join([f'{p}: {v:.2f}' for p, v in composition.items() if v > 0.01])}")
            
            # Generate events for this run
            for event_id in tqdm(range(n_events_per_run), desc=f"Run {run_id}"):
                event = self.simulate_event(run_id, event_id, condition, composition)
                all_events.append(event)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_events)
        return df
    
    def save_dataset(self, df, filename='bl4s_composition_data.csv'):
        """Save dataset to CSV file"""
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
    
    def visualize_composition_space(self, df):
        """
        Visualize the distribution of beam compositions in the dataset
        """
        # Extract composition columns
        comp_cols = [col for col in df.columns if col.startswith('comp_')]
        particle_types = [col.split('_', 1)[1] for col in comp_cols]
        
        # Group by run_id and get unique compositions
        unique_runs = df.drop_duplicates('run_id')
        
        # Create a scatter matrix for the main particle types
        main_particles = ['proton', 'electron', 'pion+', 'pion-', 'kaon+', 'muon+']
        
        # Create figure
        fig, axes = plt.subplots(len(main_particles), len(main_particles), figsize=(15, 15))
        
        # Fill scatter matrix
        for i, p1 in enumerate(main_particles):
            for j, p2 in enumerate(main_particles):
                ax = axes[i, j]
                
                if i == j:  # Diagonal: histogram
                    ax.hist(unique_runs[f'comp_{p1}'], bins=20, alpha=0.7)
                    ax.set_title(f'Distribution of {p1}', fontsize=8)
                else:  # Off-diagonal: scatter plot
                    ax.scatter(unique_runs[f'comp_{p2}'], unique_runs[f'comp_{p1}'], 
                              alpha=0.7, s=30)
                    
                    # Add regression line
                    if len(unique_runs) > 5:
                        try:
                            z = np.polyfit(unique_runs[f'comp_{p2}'], unique_runs[f'comp_{p1}'], 1)
                            p = np.poly1d(z)
                            x_range = np.linspace(min(unique_runs[f'comp_{p2}']), 
                                               max(unique_runs[f'comp_{p2}']), 10)
                            ax.plot(x_range, p(x_range), "r--", alpha=0.3)
                        except:
                            pass
                
                # Set labels only on the edges
                if i == len(main_particles) - 1:
                    ax.set_xlabel(p2, fontsize=8)
                if j == 0:
                    ax.set_ylabel(p1, fontsize=8)
                
                # Remove ticks on inner plots to reduce clutter
                if i < len(main_particles) - 1:
                    ax.set_xticklabels([])
                if j > 0:
                    ax.set_yticklabels([])
        
        plt.tight_layout()
        plt.suptitle('Beam Composition Distribution', fontsize=16, y=1.02)
        return fig
    
    def visualize_halo_response_by_composition(self, df):
        """
        Visualize how halo detector responses vary with beam composition
        """
        # First, calculate aggregate statistics per run
        # Group by run_id
        run_stats = df.groupby('run_id').agg({
            'halo_q1_hit': 'mean',
            'halo_q2_hit': 'mean',
            'halo_q3_hit': 'mean',
            'halo_q4_hit': 'mean',
            'halo_q1_adc': 'mean',
            'halo_q2_adc': 'mean',
            'halo_q3_adc': 'mean',
            'halo_q4_adc': 'mean',
            'any_halo_hit': 'mean'
        }).reset_index()
        
        # Get composition percentages for each run
        comp_cols = [col for col in df.columns if col.startswith('comp_')]
        run_comps = df.groupby('run_id')[comp_cols].first().reset_index()
        
        # Merge the stats and compositions
        run_data = pd.merge(run_stats, run_comps, on='run_id')
        
        # Create visualization
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # 1. Halo hit rates vs proton composition
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(run_data['comp_proton'], run_data['any_halo_hit'] * 100, 
                   s=80, alpha=0.7, c=run_data['comp_electron'], cmap='viridis')
        ax1.set_xlabel('Proton Fraction', fontsize=12)
        ax1.set_ylabel('Halo Hit Rate (%)', fontsize=12)
        ax1.set_title('Halo Hit Rate vs Proton Composition', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add a colorbar
        cbar = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar.set_label('Electron Fraction', fontsize=10)
        
        # 2. Average ADC values vs proton/electron composition
        ax2 = fig.add_subplot(gs[0, 1])
        total_adc = run_data[['halo_q1_adc', 'halo_q2_adc', 'halo_q3_adc', 'halo_q4_adc']].sum(axis=1)
        scatter = ax2.scatter(run_data['comp_proton'], run_data['comp_electron'], 
                            s=100, c=total_adc, cmap='plasma', alpha=0.8)
        ax2.set_xlabel('Proton Fraction', fontsize=12)
        ax2.set_ylabel('Electron Fraction', fontsize=12)
        ax2.set_title('Composition Space Colored by Total ADC', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Total ADC Value', fontsize=10)
        
        # 3. Quadrant distribution for different compositions
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Calculate quadrant asymmetry
        q_lr_asymmetry = (run_data['halo_q1_adc'] + run_data['halo_q4_adc'] - 
                          run_data['halo_q2_adc'] - run_data['halo_q3_adc']) / total_adc
        q_tb_asymmetry = (run_data['halo_q1_adc'] + run_data['halo_q2_adc'] - 
                          run_data['halo_q3_adc'] - run_data['halo_q4_adc']) / total_adc
        
        # Create scatter plot of asymmetries
        scat_asym = ax3.scatter(q_lr_asymmetry, q_tb_asymmetry, s=100, 
                              c=run_data['comp_proton'] / run_data['comp_electron'], 
                              cmap='coolwarm', alpha=0.8, norm=plt.Normalize(-2, 2))
        ax3.set_xlabel('Left-Right Asymmetry', fontsize=12)
        ax3.set_ylabel('Top-Bottom Asymmetry', fontsize=12)
        ax3.set_title('Halo Quadrant Asymmetries', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(0, color='gray', linestyle='--', alpha=0.5)
        
        cbar = plt.colorbar(scat_asym, ax=ax3)
        cbar.set_label('Proton/Electron Ratio', fontsize=10)
        
        # 4. Correlation heatmap of halo statistics vs compositions
        ax4 = fig.add_subplot(gs[1, :])
        
        # Calculate correlation matrix
        features = ['halo_q1_hit', 'halo_q2_hit', 'halo_q3_hit', 'halo_q4_hit',
                   'halo_q1_adc', 'halo_q2_adc', 'halo_q3_adc', 'halo_q4_adc',
                   'any_halo_hit']
        
        targets = comp_cols
        
        corr_data = pd.DataFrame()
        for feature in features:
            for target in targets:
                corr_data.loc[feature, target] = run_data[feature].corr(run_data[target])
        
        # Create heatmap
        sns.heatmap(corr_data, cmap='coolwarm', annot=True, fmt='.2f', ax=ax4)
        ax4.set_title('Correlation Between Halo Features and Beam Compositions', fontsize=14)
        ax4.set_xlabel('Particle Composition', fontsize=12)
        ax4.set_ylabel('Halo Detector Features', fontsize=12)
        
        plt.tight_layout()
        plt.suptitle('Beam Composition Effects on Halo Detector Response', fontsize=16, y=1.02)
        
        return fig
    
    def visualize_halo_patterns_by_composition(self, df):
        """
        Visualize halo hit patterns for different beam compositions
        """
        # Group by run_id to get unique compositions
        runs = df.drop_duplicates('run_id')
        
        # Focus on runs with significant differences in composition
        proton_rich = runs[runs['comp_proton'] > 0.5].iloc[0] if any(runs['comp_proton'] > 0.5) else None
        electron_rich = runs[runs['comp_electron'] > 0.4].iloc[0] if any(runs['comp_electron'] > 0.4) else None
        pion_rich = runs[runs['comp_pion+'] + runs['comp_pion-'] > 0.4].iloc[0] if any(runs['comp_pion+'] + runs['comp_pion-'] > 0.4) else None
        kaon_rich = runs[runs['comp_kaon+'] + runs['comp_kaon-'] > 0.25].iloc[0] if any(runs['comp_kaon+'] + runs['comp_kaon-'] > 0.25) else None
        muon_rich = runs[runs['comp_muon+'] + runs['comp_muon-'] > 0.25].iloc[0] if any(runs['comp_muon+'] + runs['comp_muon-'] > 0.25) else None
        balanced = runs[(runs['comp_proton'] < 0.3) & 
                       (runs['comp_electron'] < 0.3) & 
                       (runs['comp_pion+'] + runs['comp_pion-'] < 0.3)].iloc[0] if any((runs['comp_proton'] < 0.3) & 
                                                                                      (runs['comp_electron'] < 0.3) & 
                                                                                      (runs['comp_pion+'] + runs['comp_pion-'] < 0.3)) else None
        
        # Create a list of selected runs
        selected_runs = [r for r in [proton_rich, electron_rich, pion_rich, kaon_rich, muon_rich, balanced] if r is not None]
        
        if len(selected_runs) == 0:
            print("Not enough diverse compositions in dataset")
            return None
        
        # Create a custom colormap with CERN-like colors
        cern_blue = '#0053A1'
        cern_orange = '#FF6600'
        cern_cmap = LinearSegmentedColormap.from_list('cern_cmap', ['#f8f9fa', cern_blue, cern_orange], N=100)
        
        # Set up the figure
        n_runs = len(selected_runs)
        fig, axes = plt.subplots(1, n_runs, figsize=(n_runs * 4, 6), facecolor='#f8f9fa')
        
        if n_runs == 1:
            axes = [axes]  # Make sure axes is iterable
        
        for i, run in enumerate(selected_runs):
            # Get all events for this run
            run_events = df[df['run_id'] == run['run_id']]
            
            # Calculate mean hit rates and ADC values for each quadrant
            q1_hit_rate = run_events['halo_q1_hit'].mean()
            q2_hit_rate = run_events['halo_q2_hit'].mean()
            q3_hit_rate = run_events['halo_q3_hit'].mean()
            q4_hit_rate = run_events['halo_q4_hit'].mean()
            
            q1_adc = run_events['halo_q1_adc'].mean()
            q2_adc = run_events['halo_q2_adc'].mean()
            q3_adc = run_events['halo_q3_adc'].mean()
            q4_adc = run_events['halo_q4_adc'].mean()
            
            # Create normalized ADC values for visualization
            max_adc = max(q1_adc, q2_adc, q3_adc, q4_adc)
            if max_adc > 0:
                q1_adc_norm = q1_adc / max_adc
                q2_adc_norm = q2_adc / max_adc
                q3_adc_norm = q3_adc / max_adc
                q4_adc_norm = q4_adc / max_adc
            else:
                q1_adc_norm = q2_adc_norm = q3_adc_norm = q4_adc_norm = 0
            
            # Create a 2x2 heatmap for hit rates
            hit_heatmap = np.zeros((2, 2))
            hit_heatmap[0, 0] = q2_hit_rate  # Top-left
            hit_heatmap[0, 1] = q1_hit_rate  # Top-right
            hit_heatmap[1, 0] = q3_hit_rate  # Bottom-left
            hit_heatmap[1, 1] = q4_hit_rate  # Bottom-right
            
            adc_heatmap = np.zeros((2, 2))
            adc_heatmap[0, 0] = q2_adc_norm  # Top-left
            adc_heatmap[0, 1] = q1_adc_norm  # Top-right
            adc_heatmap[1, 0] = q3_adc_norm  # Bottom-left
            adc_heatmap[1, 1] = q4_adc_norm  # Bottom-right
            
            # Plot heatmap
            im = axes[i].imshow(adc_heatmap, cmap=cern_cmap, vmin=0, vmax=1)
            
            # Draw a circular mask to make it look like a real detector
            circle = plt.Circle((0.5, 0.5), 0.3, transform=axes[i].transAxes, 
                              fill=True, color='white', zorder=10, alpha=0.8)
            axes[i].add_patch(circle)
            axes[i].text(0.5, 0.5, 'Beam', ha='center', va='center', 
                      transform=axes[i].transAxes, fontsize=10, zorder=11)
            
            # Get composition description
            comp_desc = ", ".join([f"{p.split('_')[1]}: {v:.1%}" 
                                 for p, v in run.items() 
                                 if p.startswith('comp_') and v > 0.1])
            
            axes[i].set_title(f"Run {run['run_id']}\n{comp_desc}", fontsize=12, pad=15)
            axes[i].set_xticks([0, 1])
            axes[i].set_yticks([0, 1])
            axes[i].set_xticklabels(['Left', 'Right'])
            axes[i].set_yticklabels(['Top', 'Bottom'])
            
            # Add actual ADC values as text
            axes[i].text(0, 0, f"{q2_adc:.1f}", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
            axes[i].text(1, 0, f"{q1_adc:.1f}", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
            axes[i].text(0, 1, f"{q3_adc:.1f}", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
            axes[i].text(1, 1, f"{q4_adc:.1f}", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Normalized ADC Value', fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.suptitle('Halo Detector Patterns by Beam Composition', fontsize=16, y=0.98)
        
        return fig
    
    def create_ml_feature_extraction(self, df):
        """
        Extract and prepare features for machine learning to predict beam composition
        
        Returns DataFrame with aggregate features per run and composition targets
        """
        # List of halo features to aggregate
        halo_features = [
            'halo_q1_hit', 'halo_q2_hit', 'halo_q3_hit', 'halo_q4_hit',
            'halo_q1_adc', 'halo_q2_adc', 'halo_q3_adc', 'halo_q4_adc',
            'any_halo_hit'
        ]
        
        # List of additional features to consider
        additional_features = [
            'dwc4_x', 'dwc4_y', 'scattered_flag'
        ]
        
        # List of composition target columns
        comp_targets = [col for col in df.columns if col.startswith('comp_')]
        
        # Aggregate features by run_id
        agg_dict = {}
        
        # For each halo feature, calculate mean, std, min, max, median
        for feature in halo_features:
            agg_dict[feature + '_mean'] = (feature, 'mean')
            agg_dict[feature + '_std'] = (feature, 'std')
            agg_dict[feature + '_min'] = (feature, 'min')
            agg_dict[feature + '_max'] = (feature, 'max')
            agg_dict[feature + '_median'] = (feature, 'median')
        
        # For additional features, just calculate mean and std
        for feature in additional_features:
            if feature in df.columns:
                agg_dict[feature + '_mean'] = (feature, 'mean')
                agg_dict[feature + '_std'] = (feature, 'std')
        
        # Add more complex derived features
        df['halo_q1q3_ratio'] = df['halo_q1_adc'] / (df['halo_q3_adc'] + 1e-10)  # Avoid div by zero
        df['halo_q2q4_ratio'] = df['halo_q2_adc'] / (df['halo_q4_adc'] + 1e-10)
        df['halo_top_bottom_ratio'] = (df['halo_q1_adc'] + df['halo_q2_adc']) / (df['halo_q3_adc'] + df['halo_q4_adc'] + 1e-10)
        df['halo_left_right_ratio'] = (df['halo_q2_adc'] + df['halo_q3_adc']) / (df['halo_q1_adc'] + df['halo_q4_adc'] + 1e-10)
        df['halo_total_adc'] = df['halo_q1_adc'] + df['halo_q2_adc'] + df['halo_q3_adc'] + df['halo_q4_adc']
        
        # Add derived features to aggregation
        derived_features = [
            'halo_q1q3_ratio', 'halo_q2q4_ratio', 'halo_top_bottom_ratio', 
            'halo_left_right_ratio', 'halo_total_adc'
        ]
        
        for feature in derived_features:
            agg_dict[feature + '_mean'] = (feature, 'mean')
            agg_dict[feature + '_std'] = (feature, 'std')
        
        # Group by run_id and extract aggregate features
        agg_features = df.groupby('run_id').agg(**agg_dict)
        
        # Add composition targets (get the first row for each run, as they're all the same)
        for target in comp_targets:
            agg_features[target] = df.groupby('run_id')[target].first()
        
        # Add condition information
        agg_features['condition'] = df.groupby('run_id')['condition'].first()
        agg_features['magnet_on'] = df.groupby('run_id')['magnet_on'].first()
        
        # Add run size information
        agg_features['run_size'] = df.groupby('run_id').size()
        
        return agg_features
    
    def train_ml_model(self, df, model_type='random_forest', test_size=0.25):
        """
        Train a machine learning model to predict beam composition from halo features
        
        Parameters:
        - df: Original events dataframe
        - model_type: Type of model to train ('random_forest', 'gradient_boosting', 'neural_network')
        - test_size: Fraction of data to use for testing
        
        Returns:
        - Dictionary containing model, predictions, and evaluation metrics
        """
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        import matplotlib.gridspec as gridspec
        
        # Prepare aggregated features
        agg_features = self.create_ml_feature_extraction(df)
        
        # Get feature and target columns
        feature_cols = [col for col in agg_features.columns 
                       if not col.startswith('comp_') 
                       and col not in ['condition', 'run_size']]
        
        target_cols = [col for col in agg_features.columns if col.startswith('comp_')]
        
        # Split into features and targets
        X = agg_features[feature_cols]
        y = agg_features[target_cols]
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        
        # Select and train model
        if model_type == 'random_forest':
            model = MultiOutputRegressor(RandomForestRegressor(
                n_estimators=100, random_state=42, verbose=1))
        elif model_type == 'gradient_boosting':
            model = MultiOutputRegressor(GradientBoostingRegressor(
                n_estimators=100, random_state=42, verbose=1))
        elif model_type == 'neural_network':
            model = MLPRegressor(hidden_layer_sizes=(100, 100), 
                               max_iter=1000, random_state=42, verbose=1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Training {model_type} model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Convert to DataFrames with proper column names
        y_pred_train_df = pd.DataFrame(y_pred_train, columns=target_cols, index=y_train.index)
        y_pred_test_df = pd.DataFrame(y_pred_test, columns=target_cols, index=y_test.index)
        
        # Calculate metrics
        train_mse = {}
        test_mse = {}
        train_r2 = {}
        test_r2 = {}
        
        for i, col in enumerate(target_cols):
            train_mse[col] = mean_squared_error(y_train[col], y_pred_train_df[col])
            test_mse[col] = mean_squared_error(y_test[col], y_pred_test_df[col])
            train_r2[col] = r2_score(y_train[col], y_pred_train_df[col])
            test_r2[col] = r2_score(y_test[col], y_pred_test_df[col])
        
        # Calculate overall metrics
        overall_train_mse = np.mean(list(train_mse.values()))
        overall_test_mse = np.mean(list(test_mse.values()))
        overall_train_r2 = np.mean(list(train_r2.values()))
        overall_test_r2 = np.mean(list(test_r2.values()))
        
        # Create visualization of predictions vs true values
        plt.figure(figsize=(16, 10))
        
        gs = gridspec.GridSpec(2, 3, figure=plt.gcf())
        
        # 1. Overall prediction quality scatter plot
        ax1 = plt.subplot(gs[0, :])
        
        # Reshape data for scatter plot
        true_values = []
        predicted_values = []
        particle_types = []
        
        for col in target_cols:
            particle = col.split('_')[1]
            true_values.extend(y_test[col].values)
            predicted_values.extend(y_pred_test_df[col].values)
            particle_types.extend([particle] * len(y_test))
        
        # Create DataFrame for easier plotting
        scatter_data = pd.DataFrame({
            'True': true_values,
            'Predicted': predicted_values,
            'Particle': particle_types
        })
        
        # Plot diagonal line (perfect predictions)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # Plot scatter points
        sns.scatterplot(data=scatter_data, x='True', y='Predicted', hue='Particle', 
                      alpha=0.7, s=50, ax=ax1)
        
        ax1.set_xlabel('True Composition Fraction', fontsize=12)
        ax1.set_ylabel('Predicted Composition Fraction', fontsize=12)
        ax1.set_title(f'Prediction Quality (Test Set) - Overall R² = {overall_test_r2:.4f}', 
                    fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # 2. Individual particle R² scores
        ax2 = plt.subplot(gs[1, 0])
        
        # Get particle names without 'comp_'
        particle_names = [name.split('_')[1] for name in target_cols]
        r2_values = list(test_r2.values())
        
        # Sort by R² value
        sorted_indices = np.argsort(r2_values)
        sorted_particles = [particle_names[i] for i in sorted_indices]
        sorted_r2 = [r2_values[i] for i in sorted_indices]
        
        # Create bar chart
        bars = ax2.barh(sorted_particles, sorted_r2, alpha=0.7)
        
        # Add value labels
        for i, bar in enumerate(bars):
            ax2.text(max(0.02, bar.get_width() - 0.15), bar.get_y() + bar.get_height()/2, 
                   f'{sorted_r2[i]:.3f}', ha='center', va='center', fontsize=10, color='white',
                   fontweight='bold')
        
        ax2.set_xlabel('R² Score (Test Set)', fontsize=12)
        ax2.set_title('Prediction Quality by Particle Type', fontsize=14)
        ax2.set_xlim(0, 1)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Composition prediction examples
        ax3 = plt.subplot(gs[1, 1:])
        
        # Select a few random examples from test set
        n_examples = min(5, len(y_test))
        example_indices = np.random.choice(y_test.index, n_examples, replace=False)
        
        example_true = y_test.loc[example_indices]
        example_pred = y_pred_test_df.loc[example_indices]
        
        # Create data for grouped bar chart
        example_data = []
        
        for i, idx in enumerate(example_indices):
            for col in target_cols:
                particle = col.split('_')[1]
                example_data.append({
                    'Run': f'Run {idx}',
                    'Particle': particle,
                    'True': example_true.loc[idx, col],
                    'Predicted': example_pred.loc[idx, col]
                })
        
        example_df = pd.DataFrame(example_data)
        
        # Reshape for easier plotting
        example_melted = pd.melt(example_df, id_vars=['Run', 'Particle'], 
                               value_vars=['True', 'Predicted'],
                               var_name='Type', value_name='Fraction')
        
        # Create grouped bar chart
        sns.barplot(data=example_melted, x='Run', y='Fraction', hue='Type', 
                  palette=['#0053A1', '#FF6600'], alpha=0.8, ax=ax3, 
                  errorbar=None)
        
        # Separate by particle type
        ax3.set_title('Example Composition Predictions', fontsize=14)
        ax3.set_ylabel('Composition Fraction', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend(title='')
        
        # Ensure nice layout
        plt.tight_layout()
        
        # Create feature importance visualization if it's a RandomForest
        feature_importance = None
        if model_type == 'random_forest':
            # Extract feature importances
            importances = []
            for estimator in model.estimators_:
                importances.append(estimator.feature_importances_)
            
            # Average importances across all outputs
            mean_importances = np.mean(importances, axis=0)
            
            # Create DataFrame for visualization
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': mean_importances
            }).sort_values('Importance', ascending=False)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importance for {model_type.replace("_", " ").title()}', fontsize=16)
            
            # Plot top 20 features
            top_n = min(20, len(feature_importance))
            sns.barplot(
                data=feature_importance.head(top_n),
                x='Importance',
                y='Feature',
                palette='viridis',
                alpha=0.8
            )
            
            plt.xlabel('Average Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
        
        # Create result dictionary
        results = {
            'model': model,
            'model_type': model_type,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train_df,
            'y_pred_test': y_pred_test_df,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'overall_train_mse': overall_train_mse,
            'overall_test_mse': overall_test_mse,
            'overall_train_r2': overall_train_r2,
            'overall_test_r2': overall_test_r2,
            'feature_importance': feature_importance,
            'feature_cols': feature_cols,
            'target_cols': target_cols
        }
        
        return results
    
    def create_baseline_estimator(self, df):
        """
        Create a simple baseline estimator that predicts the average composition
        for each particle type regardless of halo features.
        
        Returns the baseline model's performance for comparison.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Prepare aggregated features 
        agg_features = self.create_ml_feature_extraction(df)
        
        # Get feature and target columns
        feature_cols = [col for col in agg_features.columns 
                       if not col.startswith('comp_') 
                       and col not in ['condition', 'run_size']]
        
        target_cols = [col for col in agg_features.columns if col.startswith('comp_')]
        
        # Split into features and targets
        X = agg_features[feature_cols]
        y = agg_features[target_cols]
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42)
        
        # Create baseline predictions (mean of each target in the training set)
        baseline_pred = {}
        for col in target_cols:
            mean_value = y_train[col].mean()
            baseline_pred[col] = [mean_value] * len(y_test)
        
        # Convert to DataFrame
        y_pred_baseline = pd.DataFrame(baseline_pred, index=y_test.index)
        
        # Calculate metrics
        baseline_mse = {}
        baseline_r2 = {}
        
        for col in target_cols:
            baseline_mse[col] = mean_squared_error(y_test[col], y_pred_baseline[col])
            baseline_r2[col] = r2_score(y_test[col], y_pred_baseline[col])
        
        # Calculate overall metrics
        overall_baseline_mse = np.mean(list(baseline_mse.values()))
        overall_baseline_r2 = np.mean(list(baseline_r2.values()))
        
        # Create result dictionary
        results = {
            'y_test': y_test,
            'y_pred_baseline': y_pred_baseline,
            'baseline_mse': baseline_mse,
            'baseline_r2': baseline_r2,
            'overall_baseline_mse': overall_baseline_mse,
            'overall_baseline_r2': overall_baseline_r2,
            'target_cols': target_cols
        }
        
        return results
    
    def create_composition_prediction_dashboard(self, df, ml_results, baseline_results=None):
        """
        Create a dashboard comparing true and predicted beam compositions
        
        Parameters:
        - df: Original events dataframe 
        - ml_results: Results from train_ml_model
        - baseline_results: Results from create_baseline_estimator (optional)
        
        Returns:
        - Matplotlib figure with prediction dashboard
        """
        # Create figure
        fig = plt.figure(figsize=(20, 12), facecolor='#f8f9fa')
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # 1. Prediction quality scatter plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Extract test data from ML results
        y_test = ml_results['y_test']
        y_pred = ml_results['y_pred_test']
        target_cols = ml_results['target_cols']
        
        # Reshape data for scatter plot
        true_values = []
        predicted_values = []
        particle_types = []
        
        for col in target_cols:
            particle = col.split('_', 1)[1]
            true_values.extend(y_test[col].values)
            predicted_values.extend(y_pred[col].values)
            particle_types.extend([particle] * len(y_test))
        
        # Create DataFrame for easier plotting
        scatter_data = pd.DataFrame({
            'True': true_values,
            'Predicted': predicted_values,
            'Particle': particle_types
        })
        
        # Plot diagonal line (perfect predictions)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # Plot scatter points
        sns.scatterplot(data=scatter_data, x='True', y='Predicted', hue='Particle', 
                      alpha=0.7, s=50, ax=ax1)
        
        ax1.set_xlabel('True Composition Fraction', fontsize=12)
        ax1.set_ylabel('Predicted Composition Fraction', fontsize=12)
        ax1.set_title(f'Prediction Quality - R² = {ml_results["overall_test_r2"]:.4f}', 
                    fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # 2. R² comparison by particle type (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Get particle names without 'comp_'
        particle_names = [name.split('_', 1)[1] for name in target_cols]
        r2_values = list(ml_results['test_r2'].values())
        
        if baseline_results is not None:
            baseline_r2_values = list(baseline_results['baseline_r2'].values())
            
            # Create DataFrame for comparison
            r2_compare = pd.DataFrame({
                'Particle': particle_names * 2,
                'R²': r2_values + baseline_r2_values,
                'Model': ['ML Model'] * len(particle_names) + ['Baseline'] * len(particle_names)
            })
            
            # Sort by ML model R² values
            sorted_indices = np.argsort(r2_values)
            sorted_particles = [particle_names[i] for i in sorted_indices]
            
            # Create grouped bar chart
            sns.barplot(x='R²', y='Particle', hue='Model', 
                      data=r2_compare, 
                      palette=['#0053A1', '#FF6600'],
                      order=sorted_particles,
                      ax=ax2)
        else:
            # Sort by R² value
            sorted_indices = np.argsort(r2_values)
            sorted_particles = [particle_names[i] for i in sorted_indices]
            sorted_r2 = [r2_values[i] for i in sorted_indices]
            
            # Create bar chart
            bars = ax2.barh(sorted_particles, sorted_r2, color='#0053A1', alpha=0.7)
            
            # Add value labels
            for i, bar in enumerate(bars):
                ax2.text(max(0.02, bar.get_width() - 0.15), bar.get_y() + bar.get_height()/2, 
                       f'{sorted_r2[i]:.3f}', ha='center', va='center', fontsize=10, color='white',
                       fontweight='bold')
        
        ax2.set_xlabel('R² Score (Test Set)', fontsize=12)
        ax2.set_title('Prediction Quality by Particle Type', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Example composition predictions (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Select a few random examples from test set
        n_examples = min(3, len(y_test))
        example_indices = np.random.choice(y_test.index, n_examples, replace=False)
        
        example_true = y_test.loc[example_indices]
        example_pred = y_pred.loc[example_indices]
        
        # Create subplots for each example
        for i, idx in enumerate(example_indices):
            # Create positions for bars
            n_particles = len(target_cols)
            width = 0.35
            positions_true = np.arange(n_particles)
            positions_pred = positions_true + width
            
            # Get values
            true_values = [example_true.loc[idx, col] for col in target_cols]
            pred_values = [example_pred.loc[idx, col] for col in target_cols]
            
            # Create subplot
            if i == 0:
                ax = ax3
            else:
                # Create a small insert for additional examples
                left = 0.05 + (i-1) * 0.48
                bottom = 0.05
                width_box = 0.4
                height_box = 0.35
                ax = fig.add_axes([left, bottom, width_box, height_box])
            
            # Plot bars
            rects1 = ax.bar(positions_true, true_values, width, label='True', color='#0053A1', alpha=0.7)
            rects2 = ax.bar(positions_pred, pred_values, width, label='Predicted', color='#FF6600', alpha=0.7)
            
            # Add labels and title
            ax.set_xlabel('Particle Type', fontsize=10 if i > 0 else 12)
            if i == 0:
                ax.set_ylabel('Composition Fraction', fontsize=12)
                ax.set_title('Example Composition Predictions', fontsize=14, fontweight='bold')
            else:
                ax.set_title(f'Run {idx}', fontsize=10)
            
            # Add x-tick labels
            ax.set_xticks(positions_true + width/2)
            ax.set_xticklabels([name.split('_', 1)[1] for name in target_cols], 
                             rotation=45 if len(target_cols) > 6 else 0,
                             fontsize=8 if i > 0 else 10)
            
            # Add legend for first plot only
            if i == 0:
                ax.legend(loc='upper right', fontsize=10)
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
        
        # 4. Feature importance or model comparison (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # If feature importance is available, show it
        if ml_results.get('feature_importance') is not None:
            feature_imp = ml_results['feature_importance']
            
            # Plot top features
            top_n = min(15, len(feature_imp))
            sns.barplot(
                data=feature_imp.head(top_n),
                x='Importance',
                y='Feature',
                palette='viridis',
                alpha=0.8,
                ax=ax4
            )
            
            ax4.set_xlabel('Feature Importance', fontsize=12)
            ax4.set_title('Top 15 Important Features', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
        else:
            # Show model comparison metrics
            if baseline_results is not None:
                metrics = {
                    'Model': ['Machine Learning', 'Baseline'],
                    'MSE': [ml_results['overall_test_mse'], baseline_results['overall_baseline_mse']],
                    'R²': [ml_results['overall_test_r2'], baseline_results['overall_baseline_r2']]
                }
                
                metrics_df = pd.DataFrame(metrics)
                
                # Create bar chart for R²
                sns.barplot(
                    data=metrics_df,
                    x='Model',
                    y='R²',
                    palette=['#0053A1', '#FF6600'],
                    alpha=0.8,
                    ax=ax4
                )
                
                # Add value labels
                for i, bar in enumerate(ax4.patches):
                    ax4.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f'{metrics_df["R²"].iloc[i]:.4f}',
                        ha='center',
                        fontsize=12,
                        fontweight='bold'
                    )
                
                ax4.set_ylim(0, max(metrics_df['R²']) * 1.2)
                ax4.set_title('Model Comparison (R²)', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3, axis='y')
            else:
                ax4.text(0.5, 0.5, 'No feature importance data available', 
                       ha='center', va='center', fontsize=14)
                ax4.set_title('Model Information', fontsize=14, fontweight='bold')
        
        # Add summary information
        model_type = ml_results.get('model_type', 'Unknown Model')
        plt.figtext(0.5, 0.01, 
                   f"Model: {model_type.replace('_', ' ').title()} | "
                   f"Overall R²: {ml_results['overall_test_r2']:.4f} | "
                   f"MSE: {ml_results['overall_test_mse']:.6f}",
                   ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.suptitle('Beam Composition Prediction Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        return fig
    
    def create_prediction_demo(self, df, ml_results, n_examples=3):
        """
        Create a demonstration of the composition prediction model
        
        This shows how the model can predict beam composition from halo detector signals
        
        Parameters:
        - df: Original events dataframe
        - ml_results: Results from train_ml_model
        - n_examples: Number of examples to show
        
        Returns:
        - Matplotlib figure with interactive demonstration
        """
        from sklearn.metrics import mean_squared_error  # Add the missing import here
        
        # Get aggregated features
        agg_features = self.create_ml_feature_extraction(df)
        
        
        # Get the model from ml_results
        model = ml_results['model']
        feature_cols = ml_results['feature_cols']
        target_cols = ml_results['target_cols']
        
        # Get a few random examples from the test set
        test_indices = ml_results['y_test'].index.tolist()
        example_indices = np.random.choice(test_indices, n_examples, replace=False)
        
        # Create a figure with subplots - one row per example
        fig, axes = plt.subplots(n_examples, 3, figsize=(18, 6 * n_examples), facecolor='#f8f9fa')
        
        if n_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(example_indices):
            # Get the example features and target values
            example_features = agg_features.loc[idx, feature_cols]
            true_composition = agg_features.loc[idx, target_cols]
            
            # Make a prediction
            example_prediction = model.predict(example_features.values.reshape(1, -1))
            example_prediction = pd.Series(example_prediction[0], index=target_cols)
            
            # 1. Visualize halo detector response
            ax1 = axes[i, 0]
            
            # Get event-level data for this run
            run_events = df[df['run_id'] == idx]
            
            # Calculate mean ADC values for each quadrant
            q1_adc = run_events['halo_q1_adc'].mean()
            q2_adc = run_events['halo_q2_adc'].mean()
            q3_adc = run_events['halo_q3_adc'].mean()
            q4_adc = run_events['halo_q4_adc'].mean()
            
            # Create normalized ADC values for visualization
            max_adc = max(q1_adc, q2_adc, q3_adc, q4_adc)
            if max_adc > 0:
                q1_adc_norm = q1_adc / max_adc
                q2_adc_norm = q2_adc / max_adc
                q3_adc_norm = q3_adc / max_adc
                q4_adc_norm = q4_adc / max_adc
            else:
                q1_adc_norm = q2_adc_norm = q3_adc_norm = q4_adc_norm = 0
            
            # Create a 2x2 heatmap for ADC values
            adc_heatmap = np.zeros((2, 2))
            adc_heatmap[0, 0] = q2_adc_norm  # Top-left
            adc_heatmap[0, 1] = q1_adc_norm  # Top-right
            adc_heatmap[1, 0] = q3_adc_norm  # Bottom-left
            adc_heatmap[1, 1] = q4_adc_norm  # Bottom-right
            
            # Create a custom colormap
            cern_blue = '#0053A1'
            cern_orange = '#FF6600'
            cern_cmap = LinearSegmentedColormap.from_list('cern_cmap', ['#f8f9fa', cern_blue, cern_orange], N=100)
            
            # Plot the heatmap
            im = ax1.imshow(adc_heatmap, cmap=cern_cmap, vmin=0, vmax=1)
            
            # Add a circular mask for the beam hole
            circle = plt.Circle((0.5, 0.5), 0.3, transform=ax1.transAxes, 
                              fill=True, color='white', zorder=10, alpha=0.8)
            ax1.add_patch(circle)
            ax1.text(0.5, 0.5, 'Beam', ha='center', va='center', 
                   transform=ax1.transAxes, fontsize=10, zorder=11)
            
            # Add quadrant labels and values
            ax1.text(0, 0, f"{q2_adc:.1f}", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
            ax1.text(1, 0, f"{q1_adc:.1f}", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
            ax1.text(0, 1, f"{q3_adc:.1f}", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
            ax1.text(1, 1, f"{q4_adc:.1f}", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
            
            ax1.set_xticks([0, 1])
            ax1.set_yticks([0, 1])
            ax1.set_xticklabels(['Left', 'Right'])
            ax1.set_yticklabels(['Top', 'Bottom'])
            
            condition = agg_features.loc[idx, 'condition']
            ax1.set_title(f"Run {idx} - {condition}\nHalo Detector Response", fontsize=12)
            
            # 2. Visualize key feature values
            ax2 = axes[i, 1]
            
            # Select some interesting features to show
            interesting_features = [
                'halo_total_adc_mean',
                'halo_top_bottom_ratio_mean',
                'halo_left_right_ratio_mean',
                'halo_q1_adc_mean',
                'halo_q2_adc_mean',
                'halo_q3_adc_mean',
                'halo_q4_adc_mean',
                'any_halo_hit_mean'
            ]
            
            # Filter for available features
            available_features = [f for f in interesting_features if f in feature_cols]
            if len(available_features) > 8:
                available_features = available_features[:8]
            
            # Get normalized feature values for visualization
            feature_values = []
            for feature in available_features:
                feature_values.append({
                    'Feature': feature.replace('_mean', '').replace('halo_', ''),
                    'Value': example_features[feature]
                })
            
            feature_df = pd.DataFrame(feature_values)
            
            # Create horizontal bar chart
            bars = ax2.barh(feature_df['Feature'], feature_df['Value'], 
                          color='#0053A1', alpha=0.7)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax2.text(width + 0.05 * max(feature_df['Value']), 
                       bar.get_y() + bar.get_height()/2, 
                       f'{width:.2f}', 
                       ha='left', va='center', fontsize=10)
            
            ax2.set_xlabel('Feature Value', fontsize=12)
            ax2.set_title('Key Features for Prediction', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='x')
            
            # 3. Compare true and predicted composition
            ax3 = axes[i, 2]
            
            # Filter for particles with significant presence
            significant_particles = [col for col in target_cols 
                                   if true_composition[col] > 0.01 or example_prediction[col] > 0.01]
            
            if len(significant_particles) > 8:
                # Keep only the top particles by true composition
                sorted_particles = sorted(significant_particles, 
                                        key=lambda x: true_composition[x], 
                                        reverse=True)
                significant_particles = sorted_particles[:8]
            
            # Extract particle names from full column names
            particle_names = [p.split('_', 1)[1] for p in significant_particles]
            
            # Prepare data for bar chart
            true_values = [true_composition[p] for p in significant_particles]
            pred_values = [example_prediction[p] for p in significant_particles]
            
            # Set width and positions
            width = 0.35
            x = np.arange(len(particle_names))
            
            # Create grouped bar chart
            ax3.bar(x - width/2, true_values, width, label='True', color='#0053A1', alpha=0.7)
            ax3.bar(x + width/2, pred_values, width, label='Predicted', color='#FF6600', alpha=0.7)
            
            # Add labels and title
            ax3.set_xlabel('Particle Type', fontsize=12)
            ax3.set_ylabel('Composition Fraction', fontsize=12)
            ax3.set_title('Beam Composition', fontsize=12)
            ax3.set_xticks(x)
            ax3.set_xticklabels(particle_names, rotation=45 if len(particle_names) > 6 else 0)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_ylim(0, max(max(true_values), max(pred_values)) * 1.1)
            
            # Add a text box with prediction quality
            mse = mean_squared_error(true_values, pred_values)
            ax3.text(0.02, 0.95, f"MSE: {mse:.4f}", transform=ax3.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.suptitle('Beam Composition Prediction from Halo Detector Signals',
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig

# Example usage
if __name__ == "__main__":
    print("BL4S Beam Halo Composition Data Generator")
    print("=========================================")
    
    # Create data generator
    print("Initializing generator...")
    generator = BL4SHaloCompositionGenerator(seed=42)
    
    # Generate dataset with multiple runs and beam compositions
    print("Generating synthetic beam composition data...")
    df = generator.generate_dataset(n_runs=30, n_events_per_run=500)
    
    # Save dataset
    print("Saving dataset...")
    generator.save_dataset(df, 'bl4s_composition_data.csv')
    
    print("\nGenerating visualizations...")
    
    # Create directory for plots
    import os
    plots_dir = "bl4s_composition_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Visualize the composition space
    print("1. Creating composition space visualization...")
    fig_comp_space = generator.visualize_composition_space(df)
    fig_comp_space.savefig(f'{plots_dir}/composition_space.png', dpi=300, bbox_inches='tight')
    
    # 2. Visualize halo response by composition
    print("2. Creating halo response by composition visualization...")
    fig_halo_comp = generator.visualize_halo_response_by_composition(df)
    fig_halo_comp.savefig(f'{plots_dir}/halo_response_by_composition.png', dpi=300, bbox_inches='tight')
    
    # 3. Visualize halo patterns by composition
    print("3. Creating halo patterns by composition visualization...")
    fig_halo_patterns = generator.visualize_halo_patterns_by_composition(df)
    if fig_halo_patterns:
        fig_halo_patterns.savefig(f'{plots_dir}/halo_patterns_by_composition.png', dpi=300, bbox_inches='tight')
    
    # 4. Create ML feature extraction
    print("4. Creating ML feature extraction...")
    ml_features = generator.create_ml_feature_extraction(df)
    
    # 5. Create baseline estimator
    print("5. Creating baseline estimator...")
    baseline_results = generator.create_baseline_estimator(df)
    
    # 6. Train ML model
    print("6. Training ML model...")
    ml_results = generator.train_ml_model(df, model_type='random_forest', test_size=0.25)
    
    # 7. Create composition prediction dashboard
    print("7. Creating composition prediction dashboard...")
    fig_pred_dashboard = generator.create_composition_prediction_dashboard(df, ml_results, baseline_results)
    fig_pred_dashboard.savefig(f'{plots_dir}/composition_prediction_dashboard.png', dpi=300, bbox_inches='tight')
    
    # 8. Create prediction demo
    print("8. Creating prediction demo...")
    fig_pred_demo = generator.create_prediction_demo(df, ml_results, n_examples=3)
    fig_pred_demo.savefig(f'{plots_dir}/prediction_demo.png', dpi=300, bbox_inches='tight')
    
    print("\nData generation and visualization complete!")
    print(f"Dataset saved as: bl4s_composition_data.csv")
    print(f"Plots saved in directory: {plots_dir}/")
