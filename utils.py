from collections import defaultdict


def get_demand_means (blood_types, default_value):
    demand_means = {k: default_value for k in blood_types}
    # Set here demand by blood type (for blood types that are different than the params['DEFAULT_VALUE_DIST'])
    demand_means['AB+'] = 3
    demand_means['B+'] = 9
    demand_means['O+'] = 18
    demand_means['B-'] = 2
    demand_means['AB-'] = 3
    demand_means['A-'] = 6
    demand_means['O-'] = 7
    demand_means['A+'] = 14

    demand_means['AB+'] = 0
    demand_means['B+'] = 0
    demand_means['O+'] = 0
    demand_means['B-'] = 0
    demand_means['AB-'] = 0
    demand_means['A-'] = 10
    demand_means['O-'] = 10
    demand_means['A+'] = 0

    demand_means['AB+'] = 3
    demand_means['B+'] = 9
    demand_means['O+'] = 18
    demand_means['B-'] = 2
    demand_means['AB-'] = 3
    demand_means['A-'] = 6
    demand_means['O-'] = 7
    demand_means['A+'] = 14

    return demand_means
    
    
def get_donation_means(blood_types, default_value):
    donation_means = {k: default_value for k in blood_types}

    # Set here donation by blood type (for blood types that are different than the params['DEFAULT_VALUE_DIST'])
    donation_means['AB+'] = 0
    donation_means['B+'] = 0
    donation_means['O+'] = 0
    donation_means['B-'] = 0
    donation_means['AB-'] = 0
    donation_means['A-'] = 10
    donation_means['O-'] = 10
    donation_means['A+'] = 0

    donation_means['AB+'] = 3
    donation_means['B+'] = 9
    donation_means['O+'] = 18
    donation_means['B-'] = 2
    donation_means['AB-'] = 3
    donation_means['A-'] = 6
    donation_means['O-'] = 7
    donation_means['A+'] = 14
    
    return donation_means

def load_params():
    params = defaultdict()
    params["epochs"] = 15
    params['blood_types'] = ['AB+', 'AB-', 'A+', 'A-', 'B+', 'B-', 'O+', 'O-']
    params["num_blood_types"] = len(params["blood_types"])
    params['allowed_transfers'] = [('AB+', 'AB+'), ('AB-', 'AB+'), ('AB-', 'AB-'), ('A+', 'AB+'), ('A+', 'A+'), ('A-', 'AB+'), ('A-', 'AB-'), ('A-', 'A+'), ('A-', 'A-'), ('B+', 'AB+'), ('B+', 'B+'), ('B-', 'AB+'), ('B-', 'AB-'), ('B-', 'B+'), ('B-', 'B-'), ('O+', 'AB+'), ('O+', 'A+'), ('O+', 'B+'), ('O+', 'O+'), ('O-', 'AB+'), ('O-', 'A+'), ('O-', 'B+'), ('O-', 'O+'), ('O-', 'AB-'), ('O-', 'A-'), ('O-', 'B-'), ('O-', 'O-')]

    # Not necessary?
    # params["blood_transfers"] = {(x, y): False for x in params['blood_types'] for y in params['blood_types']}
    # for v in params["allowed_transfers"]:
    #     params[v] = True

    params["max_age"] = 3
    params['surgery_types'] = ['urgent', 'elective']
    params['substitution'] = True

    # Set here one step contribution function parameters  - BONUSES and PENALTIES
    params["rewards"] = {}
    params["rewards"]['INFEASIABLE_SUBSTITUTION_PENALTY'] = -50
    params["rewards"]['NO_SUBSTITUTION_BONUS'] = 5
    params["rewards"]['URGENT_DEMAND_BONUS'] = 30
    params["rewards"]['ELECTIVE_DEMAND_BONUS'] = 5
    params["rewards"]['DISCARD_BLOOD_PENALTY'] = -10  # applied for the oldest age in the holding/vfa arcs

    # Set here max demand by blood type (when 'U'niform dist) or mean demand (when 'P'oisson dist)
    default_value = 20
    params['demand_means'] = get_demand_means(params["blood_types"], default_value)
    params['donation_means'] = get_donation_means(params["blood_types"], default_value)

    # Set here random surge parameters
    # params['time_periods_surge'] = set([4,8,10,12,14])
    params['time_periods_surge'] = set([i for i in range(1, params["epochs"]) if divmod(i, 3)[1] == 0])
    params['surge_prob'] = 0.7
    params['surge_factor'] = 6  # The surge demand is always going to be poisson with mean SURGE_FACTOR*demand_means, even if the regular demand distribution is Uniform
    params['surgery_types_prop']['urgent'] = 1 / 2
    params['surgery_types_prop']['elective'] = 1 - params['surgery_types_prop']['urgent']
    # Move to simulation class
    #params['blood_inventory'] = None
