import copy

def utility_position_to_reward(row, col, rewards):
    # get the reward for the position of the utility(row, column value pairs)
    if row == 1:
        return rewards[col]
    else:
        return rewards[col+3]

def utility_position_to_reward_index(row, col):
    # convert the position of the utility(row, column value pairs) to the reward 
    if row == 1:
        return col+1
    else:
        return col+4

def calculate_single_utility(row, col, rewards, utilities, gamma):
    # calculates updated utility for a single position(row, column value pairs)
    # epsilon is a float
    # rewards is a list of floats of length 6
    # utilities is a nested list of 2 lists of 3 floats each
    current_pos_reward = utility_position_to_reward(row, col, rewards)
    list_of_calculated_utilities_by_action = []
    # calculate utility for move up action
    if row == 0:
        if col == 0:
            list_of_calculated_utilities_by_action.append(0.05*utilities[row][col+1]+0.05*utilities[row][col]+0.9*utilities[row][col])
        elif col == 2:
            list_of_calculated_utilities_by_action.append(0.05*utilities[row][col-1]+0.05*utilities[row][col]+0.9*utilities[row][col])
        else:
            list_of_calculated_utilities_by_action.append(0.05*utilities[row][col-1]+0.05*utilities[row][col+1]+0.9*utilities[row][col])
    else:
        if col == 0:
            list_of_calculated_utilities_by_action.append(0.9*utilities[row-1][col]+0.05*utilities[row][col+1]+0.05*utilities[row][col])
        elif col == 2:
            list_of_calculated_utilities_by_action.append(0.9*utilities[row-1][col]+0.05*utilities[row][col-1]+0.05*utilities[row][col])
        else:
            list_of_calculated_utilities_by_action.append(0.9*utilities[row-1][col]+0.05*utilities[row][col-1]+0.05*utilities[row][col+1])
    # calculate utility for move right action
    if col == 0:
        if row == 0:
            list_of_calculated_utilities_by_action.append(0.9*utilities[row][col+1]+0.05*utilities[row+1][col]+0.05*utilities[row][col])
        else:
            list_of_calculated_utilities_by_action.append(0.9*utilities[row][col+1]+0.05*utilities[row-1][col]+0.05*utilities[row][col])
    elif col == 2:
        if row == 0:
            list_of_calculated_utilities_by_action.append(0.05*utilities[row+1][col]+0.05*utilities[row][col]+0.9*utilities[row][col])
        else:
            list_of_calculated_utilities_by_action.append(0.05*utilities[row-1][col]+0.05*utilities[row][col]+0.9*utilities[row][col])
    else:
        if row == 0:
            list_of_calculated_utilities_by_action.append(0.9*utilities[row][col+1]+0.05*utilities[row+1][col]+0.05*utilities[row][col])
        else:
            list_of_calculated_utilities_by_action.append(0.9*utilities[row][col+1]+0.05*utilities[row-1][col]+0.05*utilities[row][col])
    # calculate utility for move down action
    if row == 1:
        if col == 0:
            list_of_calculated_utilities_by_action.append(0.05*utilities[row][col+1]+0.9*utilities[row][col]+0.05*utilities[row][col])
        elif col == 2:
            list_of_calculated_utilities_by_action.append(0.05*utilities[row][col-1]+0.9*utilities[row][col]+0.05*utilities[row][col])
        else:
            list_of_calculated_utilities_by_action.append(0.05*utilities[row][col-1]+0.05*utilities[row][col+1]+0.9*utilities[row][col])
    else:
        if col == 0:
            list_of_calculated_utilities_by_action.append(0.9*utilities[row+1][col]+0.05*utilities[row][col+1]+0.05*utilities[row][col])
        elif col == 2:
            list_of_calculated_utilities_by_action.append(0.9*utilities[row+1][col]+0.05*utilities[row][col-1]+0.05*utilities[row][col])
        else:
            list_of_calculated_utilities_by_action.append(0.9*utilities[row+1][col]+0.05*utilities[row][col-1]+0.05*utilities[row][col+1])
    # calculate utility for move left action
    if col == 0:
        if row == 0:
            list_of_calculated_utilities_by_action.append(0.05*utilities[row+1][col]+0.05*utilities[row][col]+0.9*utilities[row][col])
        else:
            list_of_calculated_utilities_by_action.append(0.05*utilities[row-1][col]+0.05*utilities[row][col]+0.9*utilities[row][col])
    elif col == 2:
        if row == 0:
            list_of_calculated_utilities_by_action.append(0.9*utilities[row][col-1]+0.05*utilities[row+1][col]+0.05*utilities[row][col])
        else:
            list_of_calculated_utilities_by_action.append(0.9*utilities[row][col-1]+0.05*utilities[row-1][col]+0.05*utilities[row][col])
    else:
        if row == 0:
            list_of_calculated_utilities_by_action.append(0.9*utilities[row][col-1]+0.05*utilities[row+1][col]+0.05*utilities[row][col])
        else:
            list_of_calculated_utilities_by_action.append(0.9*utilities[row][col-1]+0.05*utilities[row-1][col]+0.05*utilities[row][col])

    # calculate utility for do nothing action
    list_of_calculated_utilities_by_action.append(1*utilities[row][col])

    # calculate the max utility out of calculated utilities for all actions
    max_utility = max(list_of_calculated_utilities_by_action)
    # calculate the single utility for the current position
    single_utility = current_pos_reward + gamma * max_utility
    return single_utility

def calculate_updated_utility_for_single_iteration(rewards, utilities, gamma=0.999):
    # update utilities of all indeces for a single iteration
    # rewards is a list of floats of length 6
    # utilities is a nested list of 2 lists of 3 floats each
    for row in range(1,-1,-1):
        for col in range(3):
            if row == 1 and col == 2:
                # terminal state is assigned a constant utility of 1 to expedite convergence
                utilities[row][col] = 1
                continue
            # calculate and update utility for other states
            utilities[row][col] = calculate_single_utility(row, col, rewards, utilities, gamma)
    return utilities

def calculate_utility_maginutude(utilities):
    # calculate the utility maginutude of the difference between 2 utilities
    # utilities is a nested list of 2 lists of 3 floats each
    # returns a float
    utility_maginutude = 0
    for col in range(3):
        for row in range(2):
            utility_maginutude += abs(utilities[row][col])
    return utility_maginutude

def flatten_utilities(utilities):
    # flatten the nested list of utilities to a single list
    # utilities is a nested list of 2 lists of 3 floats each
    # returns a list of 6 floats
    flattened_utilities = []
    for row in range(1,-1,-1):
        for col in range(3):
            flattened_utilities.append(utilities[row][col])
    return flattened_utilities

def get_state_utilities(epsilon, rewards, gamma=0.999):
    # main function to calculate utilities for all states
    # epsilon is a float
    # rewards is a list of floats of length 6
    expected_utilities = [[0 for _ in range(3)] for _ in range(2)]
    min_delta_value = epsilon * (1 - gamma) / gamma
    delta = epsilon+1; iteration = 1; utility_maginutude = 0
    while delta > min_delta_value:
        delta = 0
        updated_utilities = calculate_updated_utility_for_single_iteration(rewards, copy.deepcopy(expected_utilities), gamma)
        difference = [[updated_utilities[row][col] - expected_utilities[row][col] for col in range(3)] for row in range(2)]
        utility_maginutude = calculate_utility_maginutude(difference)
        delta = max(utility_maginutude, delta)
        iteration += 1
        expected_utilities = updated_utilities
    return flatten_utilities(expected_utilities)


print(get_state_utilities(0.1, [-0.1, -0.1, 1, -0.1, -0.1, -0.05]))

