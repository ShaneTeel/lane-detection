import numpy as np
import copy

class RANSACConfigManager():
    '''Validates user-passed configs, and merges with default configs if appropriate'''

    _VALID_RANSAC_SETUP = {
        "generator": {
            'filter': {'filter_type': ['median', 'mean'], 'n_std': np.arange(0.0, 3.1, 0.01).tolist()},
            'polyfit': {'n_iter': list(range(1, 101)), 'degree': [1, 2, 3], 'threshold': list(range(0, 101)), 'min_inliers': np.arange(0.0, 1.00, 0.01), 'weight': list(range(1, 11)), "factor": np.arange(0.0, 1.00, 0.01)},
        },
        "preprocessor": {
            'in_range': {'lower_bounds': list(range(0, 255)), 'upper_bounds': list(range(1, 256))},
            'canny': {'weak_edge': list(range(0, 301)), 'sure_edge': list(range(0, 301)), 'blur_ksize': list(range(3, 16, 2)), "blur_order": ["before", "after"]},
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
        # CHECK 1: Does the user-passed configuration include invalid events (outer-most keys)
        invalid_keys = set(self.final_configs.keys()).difference(self._VALID_RANSAC_SETUP.keys())
        if len(invalid_keys) > 0:
            raise KeyError(f"ERROR: User-passed configuration includes an invalid keyword argument {invalid_keys}")
        
        # Iterate through validated events (outer-most keys)
        for event in self.final_configs.keys():
            
            # CHECK 1: If proposed event in valid events
            for step in self.final_configs[event].keys():
                if event not in self._VALID_RANSAC_SETUP[event].keys(): 

                # If not, raise KeyError
                raise KeyError(f"ERROR: User configuration includes invalid step: '{step}\'")
           
            else:
                # If all outer keys validated, extract inner keys
                proprosed_args = [arg for arg in self.final_configs.get(step)] 
                proposed_vals = [val for val in self.final_configs.get(step).values()]
                valid_vals = [val for val in self._VALID_RANSAC_SETUP.get(step).values()]
                
                # Iterate through inner keys
                for i, arg in enumerate(proprosed_args): 

                    # Extract valid inner keys
                    valid_args = [arg for arg in self._VALID_RANSAC_SETUP.get(step)] 
                    
                    # CHECK 2: If inner key from final_configs is in valid_configs
                    if arg not in valid_args: 

                        # If not, raise KeyError
                        raise KeyError(f"ERROR: User configuration includes invalid keyword argument ('{arg}') for step '{step}\'") 
                    
                    if isinstance(valid_vals[i][0], int) or isinstance(valid_vals[i][0], float): 
                        
                        if proposed_vals[i] in valid_vals[i]:
                            print(f"Successfully validated {step} {arg} {proposed_vals[i]}")
                            continue
                        
                        else:
                            raise ValueError(f"ERROR: Argument passed to '{step}', '{arg}' not in valid range of {min(valid_vals[i])} - {max(valid_vals[i])}") 
                    
                    else:
                       
                        if proposed_vals[i] in valid_vals[i]:
                            print(f"Successfully validated {step} {arg} {proposed_vals[i]}")
                            continue
                        
                        else:
                            raise ValueError(f"ERROR: Argument passed to '{step}', '{arg}' not one of {valid_vals[i]}")