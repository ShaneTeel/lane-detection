import copy

class ConfigManager():
    '''Validates user-passed configs, and merges with default configs if appropriate'''

    def __init__(self, user_configs:dict, default_configs:dict, valid_config_setup:dict):

        self.user_configs:dict = user_configs
        self.final_configs: dict = copy.deepcopy(default_configs)
        self.valid_config_setup = valid_config_setup

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
        invalid_events = set(self.final_configs.keys()).difference(self.valid_config_setup.keys())
        if len(invalid_events) > 0:

            # If not, raise KeyError
            raise KeyError(f"ERROR: User-passed configuration includes an invalid keyword argument {invalid_events}")
         
        # Iterate through validated events (Level 1 Keys)
        for event in self.final_configs.keys():

            # CHECK 2: If proposed step in valid steps
            invalid_steps = set(self.final_configs[event].keys()).difference(self.valid_config_setup[event].keys())
            if len(invalid_steps) > 0:
        
                # If not, raise KeyError
                raise KeyError(f"ERROR: User configuration includes invalid step/s: {invalid_steps}")
            
            # Iterate through validated steps (Level 2 Keys)
            for step in self.final_configs[event].keys():
            
                # CHECK 2: If proposed step in valid steps
                invalid_params = set(self.final_configs[event][step].keys()).difference(self.valid_config_setup[event][step].keys())
                if len(invalid_params) > 0:

                    # If not, raise KeyError
                    raise KeyError(f"ERROR: User configuration includes the following invalid keywords: {invalid_params}")

                # Iterate through validated parameters (Level 3 Keys)
                for param in self.final_configs[event][step].keys():
                    
                    # Extract proposed arguments for each param
                    proposed_arg = self.final_configs[event][step][param]

                    # Extract valid arguments for each param
                    valid_args = self.valid_config_setup[event][step][param]


        
                    # If valid arg of type int or float
                    if valid_args is None:
                        continue
                    
                    if isinstance(valid_args[0], int) or isinstance(valid_args[0], float):
                        
                        min_valid = valid_args[0]
                        max_valid = valid_args[-1]

                        # Check 3A: If proposed arg in valid arg
                        if not min_valid <= proposed_arg <= max_valid:

                            # If not, raise ValueError
                            condition = f"{min_valid} - {max_valid}"
                            raise ValueError(f"ERROR: User configuration includes invalid argument ({proposed_arg}) for parameter {param} in step {step} of event {event}.\n\t\tMust be of type {type(min_valid)} and in the range of {condition}.")
                
                    # If valid arg of type str
                    if isinstance(valid_args[0], str):

                        # Check 3B: If proposed arg in valid arg
                        if proposed_arg not in valid_args:
                            raise ValueError(f"ERROR: User configuration includes invalid argument ({proposed_arg}) for parameter {param} in step {step} of event {event}.\n\t\tMust be one of {valid_args}.")