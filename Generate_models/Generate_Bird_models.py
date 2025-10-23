import numpy as np
import math

def Generate(S,N,A,RSeed,subfolder = ""):
    O = 2
    np.random.seed(RSeed)

    expert_actions = dict()
    actions = [a for a in range(A)]
    max_per = math.factorial(A)
    while len(expert_actions) < N:
        expert = tuple([tuple(np.random.permutation(actions)) for _ in range(S)])
        if expert not in expert_actions.values() or len(expert_actions) >= max_per:
            expert_actions[len(expert_actions)] = expert
    
    probabilities = [round(0.05*i,2) for i in range(21)]
    action_list = []
    if S > 2:
        for _ in range(A):
            a1 = np.random.randint(0,21)
            a2 = np.random.randint(0,21-a1)
            action_list.append([probabilities[a1], probabilities[a2], probabilities[20-a1-a2]])
    else:
        for _ in range(A):
            a1 = np.random.randint(0,21)
            action_list.append([probabilities[a1], probabilities[20-a1]])
    action_list.sort()
    action_list = action_list[::-1]

    expert_observations = dict()
    if S > 2:
        while len(expert_observations) < N:
            obs_list = []
            for s in range(S-2):
                o1 = np.random.randint(0,21)
                obs_list.append(tuple([probabilities[o1],probabilities[20-o1]]))
            obs_list.sort()
            expert = tuple(obs_list[::-1])
            if expert not in expert_observations.values():
                expert_observations[len(expert_observations)] = expert

    states = ", ".join([f"s{s}" for s in range(S)])
    state_string = f"{S}, [{states}]\n"
    
    envs = ", ".join([f"e{n}" for n in range(N)])
    env_string = f"{N}, [{envs}]\n"

    actions = ", ".join([f"a{a}" for a in range(A)])
    action_string = f"{A}, [{actions}]\n"
    
    obs = ", ".join([f"o{o}" for o in range(O)])
    obs_string = f"{O}, [{obs}]\n\n"

    transition_strings = ["# Transition function (s,a,s -> p)"]
    single_transition = ["# Transition function (s,a,s -> p)"]
    if S > 2:
        for si in range(S):
            for a in range(A):
                if si == 0:
                    transition_strings.append(f"{si},{a},{si} -> pt_{si}.{a}.0")
                    transition_strings.append(f"{si},{a},{si+1} -> pt_{si}.{a}.1")
                    transition_strings.append(f"{si},{a},{si+2} -> pt_{si}.{a}.2")
                    single_transition.append(f"{si},{a},{si} -> {action_list[expert_actions[0][si][a]][0]}")
                    single_transition.append(f"{si},{a},{si+1} -> {action_list[expert_actions[0][si][a]][1]}")
                    single_transition.append(f"{si},{a},{si+2} -> {action_list[expert_actions[0][si][a]][2]}")
                elif si == S-1:
                    transition_strings.append(f"{si},{a},{si-2} -> pt_{si}.{a}.0")
                    transition_strings.append(f"{si},{a},{si-1} -> pt_{si}.{a}.1")
                    transition_strings.append(f"{si},{a},{si} -> pt_{si}.{a}.2")
                    single_transition.append(f"{si},{a},{si-2} -> {action_list[expert_actions[0][si][a]][0]}")
                    single_transition.append(f"{si},{a},{si-1} -> {action_list[expert_actions[0][si][a]][1]}")
                    single_transition.append(f"{si},{a},{si} -> {action_list[expert_actions[0][si][a]][2]}")
                else:
                    transition_strings.append(f"{si},{a},{si-1} -> pt_{si}.{a}.0")
                    transition_strings.append(f"{si},{a},{si} -> pt_{si}.{a}.1")
                    transition_strings.append(f"{si},{a},{si+1} -> pt_{si}.{a}.2")
                    single_transition.append(f"{si},{a},{si-1} -> {action_list[expert_actions[0][si][a]][0]}")
                    single_transition.append(f"{si},{a},{si} -> {action_list[expert_actions[0][si][a]][1]}")
                    single_transition.append(f"{si},{a},{si+1} -> {action_list[expert_actions[0][si][a]][2]}")
    else:
        for si in range(S):
            for a in range(A):
                if si == 0:
                    transition_strings.append(f"{si},{a},{si} -> pt_{si}.{a}.0")
                    transition_strings.append(f"{si},{a},{si+1} -> pt_{si}.{a}.1")
                    single_transition.append(f"{si},{a},{si} -> {action_list[expert_actions[0][si][a]][0]}")
                    single_transition.append(f"{si},{a},{si+1} -> {action_list[expert_actions[0][si][a]][1]}")
                else:
                    transition_strings.append(f"{si},{a},{si-1} -> pt_{si}.{a}.0")
                    transition_strings.append(f"{si},{a},{si} -> pt_{si}.{a}.1")
                    single_transition.append(f"{si},{a},{si-1} -> {action_list[expert_actions[0][si][a]][0]}")
                    single_transition.append(f"{si},{a},{si} -> {action_list[expert_actions[0][si][a]][1]}")
                    

    observation_strings = ["# Observation function (a,s,o -> p)"]
    single_observation = ["# Observation function (a,s,o -> p)"]
    for si in range(S):
        if si == 0:
            observation_strings.append(f"_,{si},0 -> 1")
            single_observation.append(f"_,{si},0 -> 1")
        elif si == S-1:
            observation_strings.append(f"_,{si},1 -> 1")
            single_observation.append(f"_,{si},1 -> 1")
        elif S > 2:
            observation_strings.append(f"_,{si},0 -> pt_{si}.0")
            observation_strings.append(f"_,{si},1 -> pt_{si}.1")
            single_observation.append(f"_,{si},0 -> {expert_observations[0][si-1][0]}")
            single_observation.append(f"_,{si},1 -> {expert_observations[0][si-1][1]}")
    
    reward_strings = ["# Reward function (s,a -> r)"]
    for si in range(S):
        for a in range(A):
            rew = si*5
            if a > 0:
                rew += -5
            reward_strings.append(f"{si},{a} -> {rew}")

    belief_strings = ["# Initial beliefs (s -> p)"]
    belief_strings.append("0 -> 1")
    
    trns_par_list = [f"pt_{si}.{a}.{i}" for si in range(S) for a in range(A) for i in range(min(S,3))]
    obs_par_list = [f"pt_{si+1}.{i}" for si in range(S-2) for i in range(min(S,2))]
    obs_par_strings = []
    trns_par_strings = []
    trns_obs_par_strings = []
    for n in range(N):
        transition_pars = [f"{action_list[expert_actions[n][si][a]][i]}" for si in range(S) for a in range(A) for i in range(min(S,3))]
        transition_pars = ", ".join(transition_pars)

        observation_pars = [f"{expert_observations[n][si][i]}" for si in range(S-2) for i in range(2)]
        observation_pars = ", ".join(observation_pars)

        obs_par_strings.append(f"[{observation_pars}]")
        trns_par_strings.append(f"[{transition_pars}]")
        trns_obs_par_strings.append(f"[{transition_pars}, {observation_pars}]")

    
    filename = f"Models/{subfolder}Birds_MEPOMDP_S{S}_N{N}_A{A}_R{RSeed}.txt"
    pars = ", ".join(trns_par_list+obs_par_list)
    par_strings = [f"{len(trns_par_list) + len(obs_par_list)}, [{pars}]"] + trns_obs_par_strings
    with open(filename, "w") as file:
        file.write(state_string)
        file.write(env_string)
        file.write(action_string)
        file.write(obs_string)
        file.write("\n".join(par_strings)+"\n\n")
        file.write("\n".join(transition_strings)+"\n\n")
        file.write("\n".join(observation_strings)+"\n\n")
        file.write("\n".join(reward_strings)+"\n\n")
        file.write("\n".join(belief_strings))

    filename = f"Models/{subfolder}Birds_POMEMDP_S{S}_N{N}_A{A}_R{RSeed}.txt"
    pars = ", ".join(trns_par_list)
    par_strings = [f"{len(trns_par_list)}, [{pars}]"] + trns_par_strings
    with open(filename, "w") as file:
        file.write(state_string)
        file.write(env_string)
        file.write(action_string)
        file.write(obs_string)
        file.write("\n".join(par_strings)+"\n\n")
        file.write("\n".join(transition_strings)+"\n\n")
        file.write("\n".join(single_observation)+"\n\n")
        file.write("\n".join(reward_strings)+"\n\n")
        file.write("\n".join(belief_strings))
    
    filename = f"Models/{subfolder}Birds_MOPOMDP_S{S}_N{N}_A{A}_R{RSeed}.txt"
    pars = ", ".join(obs_par_list)
    par_strings = [f"{len(obs_par_list)}, [{pars}]"] + obs_par_strings
    with open(filename, "w") as file:
        file.write(state_string)
        file.write(env_string)
        file.write(action_string)
        file.write(obs_string)
        file.write("\n".join(par_strings)+"\n\n")
        file.write("\n".join(single_transition)+"\n\n")
        file.write("\n".join(observation_strings)+"\n\n")
        file.write("\n".join(reward_strings)+"\n\n")
        file.write("\n".join(belief_strings))

# Generate a ME-POMDP, PO-MEMDP, and MO-POMDP for the Bird problem with s>=2 states, n experts, and a actions, using random seed r to create the observation and transition probabilities and the experts.
if __name__ == "__main__":
    s,n,a = 3,3,3
    for r in range(100):
        Generate(s,n,a,r,"Triviality_test/")