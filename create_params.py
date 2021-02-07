import os

def update_single_envirement_var(var_name, value):
    '''
    updates a single envirement variable
    
    inputs:
        var_name: str, name of envirement variable to create
        value: str, value to make new envirement variable
    returns:
        None
    '''
    os.environ[var_name] = value
    return None

def update_env_variables(dict_):
    '''
    updates envirement variables from a dictionary
    
    inputs:
        dict_: dict, key value pairs that will be updated as env variables
        each value should be a str
    returns:
        None
    '''
    for key in dict_:
        update_single_envirement_var(key, dict_[key])
    return None

def get_env_variable(var):
    '''
    retreives single envirement variable
    
    inputs:
        var: str, variable key to be retreived
    
    returns:
        str, value of var envirement variable key
    raises:
        KeyError: if var does not exist as an envirement variable
    '''
    return os.environ[var]

def update_params_dict(params_dict):
    '''
    updates params dictionary with values stored as envirement variables
    
    inputs:
        params_dict: dict, dictionary whose values will be updated
    
    returns:
        params_dict as modified as described above, values will remain
        unchanged if no variable exists as an envirement variable
    '''
    for key in params_dict:
        try:
            new_val = get_env_variable(key)
            params_dict[key] = new_val
        except KeyError:
            pass
        
    return params_dict
    