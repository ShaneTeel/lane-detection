import numpy as np
import copy

class RANSACConfigManager():
    '''Validates user-passed configs, and merges with default configs if appropriate'''

    _VALID_RANSAC_SETUP = {
        "preprocessor": {
            'in_range': {'lower_bounds': list(range(0, 255)), 'upper_bounds': list(range(1, 256))},
            'canny': {'weak_edge': list(range(0, 301)), 'sure_edge': list(range(0, 301)), 'blur_ksize': list(range(3, 16, 2)), "blur_order": ["before", "after"]},
        },        
        "generator": {
            'filter': {'filter_type': ['median', 'mean'], 'n_std': np.arange(0.0, 3.1, 0.01).tolist()},
            'polyfit': {'n_iter': list(range(1, 101)), 'degree': [1, 2, 3], 'threshold': list(range(0, 101)), 'min_inliers': np.arange(0.0, 1.00, 0.01), 'weight': list(range(1, 11)), "factor": np.arange(0.0, 1.00, 0.01)},
        }
    }

    _DEFAULT_RANSAC_CONFIGS = {
        "preprocessor": {
            'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
            'canny': {'weak_edge': 50, 'sure_edge': 100, 'blur_ksize': 3, "blur_order": "before"},
        },
        "generator": {
            'filter': {'filter_type': 'median', 'n_std': 2}, 
            'polyfit': {'n_iter': 100, 'degree': 2, 'threshold': 50, 'min_inliers': 0.6, 'weight': 5, 'factor': 0.1}
        }
    }

    def __init__(self, user_configs:dict):

        self.user_configs:dict = user_configs
        self.final_configs:dict = copy.deepcopy(self._DEFAULT_RANSAC_CONFIGS)

    def manage(self):
        if self.user_configs is None:
            return self.final_configs
        
        self._merge_configs()
        self._validate_configs()
    
        return self.final_configs

    def _merge_configs(self):
        def _recursive_update(default_dict:dict, new_dict:dict):
            for key, val in new_dict.items():
                if isinstance(val, dict) and key in default_dict and isinstance(default_dict[key], dict):
                    _recursive_update(default_dict[key], val)
                else:
                    default_dict[key] = val

        _recursive_update(self.final_configs, self.user_configs)
        
    def _validate_configs(self):
        # CHECK 1: Does the user-passed configuration include invalid events (Level 1 Keys)
        invalid_events = set(self.final_configs.keys()).difference(self._VALID_RANSAC_SETUP.keys())
        if len(invalid_events) > 0:

            # If not, raise KeyError
            raise KeyError(f"ERROR: User-passed configuration includes an invalid keyword argument {invalid_events}")
         
        # Iterate through validated events (Level 1 Keys)
        for event in self.final_configs.keys():

            # CHECK 2: If proposed step in valid steps
            invalid_steps = set(self.final_configs[event].keys()).difference(self._VALID_RANSAC_SETUP[event].keys())
            if len(invalid_steps) > 0:
        
                # If not, raise KeyError
                raise KeyError(f"ERROR: User configuration includes invalid step/s: {invalid_steps}")
            
            # Iterate through validated steps (Level 2 Keys)
            for step in self.final_configs[event].keys():
            
                # CHECK 2: If proposed step in valid steps
                invalid_params = set(self.final_configs[event][step].keys()).difference(self._VALID_RANSAC_SETUP[event][step].keys())
                if len(invalid_params) > 0:

                    # If not, raise KeyError
                    raise KeyError(f"ERROR: User configuration includes invalid parameter/s: {invalid_params}")

                # Iterate through validated parameters (Level 3 Keys)
                for param in self.final_configs[event][step].keys():
                    
                    # Extract proposed arguments for each param
                    proposed_arg = self.final_configs[event][step][param]

                    # Extract valid arguments for each param
                    valid_args = self._VALID_RANSAC_SETUP[event][step][param]
                    
        
                    # If valid arg of type int or float
                    if isinstance(valid_args[0], int) or isinstance(valid_args[0], float):
                        
                        # Check 3A: If proposed arg in valid arg
                        if proposed_arg not in valid_args:

                            # If not, raise ValueError
                            condition = f"{min(valid_args)} - {max(valid_args)}" if len(valid_args) > 10 else f"{valid_args}"
                            raise ValueError(f"ERROR: User configuration includes invalid argument ({proposed_arg}) for parameter {param} in step {step} of event {event}.\n\t\tMust be of type {type(valid_args[0])} and in the range of {condition}.")
                
                    # If valid arg of type str
                    if isinstance(valid_args[0], str):

                        # Check 3B: If proposed arg in valid arg
                        if proposed_arg not in valid_args:
                            raise ValueError(f"ERROR: User configuration includes invalid argument ({proposed_arg}) for parameter {param} in step {step} of event {event}.\n\t\tMust be one of {valid_args}.")