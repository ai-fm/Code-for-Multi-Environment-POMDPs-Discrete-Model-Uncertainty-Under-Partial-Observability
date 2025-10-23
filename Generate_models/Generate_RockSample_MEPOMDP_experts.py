import random
from itertools import product
import itertools
import numpy as np
import math

def state(si,sj,gr,N,G):
    return si * N*pow(2,G) + sj*pow(2,G) + gr

def status_gr(g_rocks,k,gr,G):
    good = False
    gr_bin = format(gr, f'0{G}b')
    for g in range(G):
        if gr_bin[g] == '1':
            if g_rocks[g] == k:
                good = True
                break
    return good

def status_g(g_rocks,k,g):
    return g_rocks[g] == k

def Generate(filename,N,G,K,RSeed):
    random.seed(RSeed)
    rockPositions = dict()
    while len(rockPositions) < K:
        rock = (random.randint(0,N-1),random.randint(0,N-1))
        if rock != (0,0) and rock not in rockPositions.values():
            rockPositions[len(rockPositions)] = rock
    
    goodRocks = []
    for comb in itertools.combinations([i for i in range(K)], G):
        goodRocks.append(list(comb))

    rocks = [".".join(list(tup)) for tup in list(product(*[["b", "g"] for _ in range(G)]))]
    S = N*N*pow(2,G)+1
    states = ", ".join([f"s{i}.{j}-{r}" for i in range(N) for j in range(N) for r in rocks])
    state_string = f"{S}, [{states}, exit]\n"
    
    env_string = "1, [e1]\n"
    A = K+5
    check_actions = ", ".join([f"Check{k}" for k in range(K)])
    action_string = f"{A}, [North, South, East, West, Sample, {check_actions}]\n"
    obs_string = "3, [Bad, Good, Nothing]\n\n"

    all_transition_strings = []
    transition_strings = ["# Transition function (s,a,s -> p)"]
    for expert in range(len(goodRocks)):
        for si in range(N):
            for sj in range(N):
                for gr in range(pow(2,G)):
                    if si < N-1:
                        transition_strings.append(f"{state(si,sj,gr,N,G)},0,{state(si+1,sj,gr,N,G)} -> 1")
                    else:
                        transition_strings.append(f"{state(si,sj,gr,N,G)},0,{state(si,sj,gr,N,G)} -> 1")
                    if si > 0:
                        transition_strings.append(f"{state(si,sj,gr,N,G)},1,{state(si-1,sj,gr,N,G)} -> 1")
                    else:
                        transition_strings.append(f"{state(si,sj,gr,N,G)},1,{state(si,sj,gr,N,G)} -> 1")
                    if sj < N-1:
                        transition_strings.append(f"{state(si,sj,gr,N,G)},2,{state(si,sj+1,gr,N,G)} -> 1")
                    else:
                        transition_strings.append(f"{state(si,sj,gr,N,G)},2,{S-1} -> 1")
                    if sj > 0:
                        transition_strings.append(f"{state(si,sj,gr,N,G)},3,{state(si,sj-1,gr,N,G)} -> 1")
                    else:
                        transition_strings.append(f"{state(si,sj,gr,N,G)},3,{state(si,sj,gr,N,G)} -> 1")
                if (si,sj) in rockPositions.values():
                    k = [key for key, value in rockPositions.items() if value == (si,sj)][0]
                    for gr in range(pow(2,G)):
                        rest_prob = 1
                        gr_bin = format(gr, f'0{G}b')
                        for g in range(G):
                            if gr_bin[g] == '1':
                                gr_bin = gr_bin[0:g]+"0"+gr_bin[g+1:]
                                gr2 = int(gr_bin,2)
                                pr = "1" if status_g(goodRocks[expert],k,g) else "0"
                                transition_strings.append(f"{state(si,sj,gr,N,G)},4,{state(si,sj,gr2,N,G)} -> {pr}")
                                rest_prob += -status_g(goodRocks[expert],k,g)
                        transition_strings.append(f"{state(si,sj,gr,N,G)},4,{state(si,sj,gr,N,G)} -> {rest_prob}")
                for k in range(K):
                    a = k+5
                    for gr in range(pow(2,G)):
                        transition_strings.append(f"{state(si,sj,gr,N,G)},{a},{state(si,sj,gr,N,G)} -> 1")
        transition_strings.append(f"{S-1},_,{S-1} -> 1")
        all_transition_strings.append("\n".join(transition_strings)+"\n\n")
        transition_strings = ["# Transition function (s,a,s -> p)"]

    half_dist = math.floor(N/2)
    eff_lev = dict()
    all_observation_strings = []
    observation_strings = ["# Observation function (a,s,o -> p)"]
    for expert in range(len(goodRocks)):
        for a in range(5):
            for si in range(N):
                for sj in range(N):
                    for gr in range(pow(2,G)):
                        observation_strings.append(f"{a},{state(si,sj,gr,N,G)},2 -> 1")
        for k in range(K):
            a = k+5
            for si in range(N):
                for sj in range(N):
                    (sii,sjj) = rockPositions[k]
                    dist = np.linalg.norm([si-sii,sj-sjj])
                    eff = round(pow(2,-dist/half_dist),6)
                    c_eff = len(eff_lev)
                    if eff not in eff_lev.values():
                        eff_lev[c_eff] = eff
                    else:
                        c_eff = [key for key, value in eff_lev.items() if value == eff][0]
                    for gr in range(pow(2,G)):
                        gr_bin = format(gr, f'0{G}b')
                        pr = round(0.5 + 0.5*eff_lev[c_eff],6) if status_gr(goodRocks[expert],k,gr,G) else round(1 - (0.5 + 0.5*eff_lev[c_eff]),6)
                        observation_strings.append(f"{a},{state(si,sj,gr,N,G)},1 -> {pr}")
                        observation_strings.append(f"{a},{state(si,sj,gr,N,G)},0 -> {round(1-pr,6)}")
        observation_strings.append(f"_,{S-1},2 -> 1")
        all_observation_strings.append("\n".join(observation_strings)+"\n\n")
        observation_strings = ["# Observation function (a,s,o -> p)"]

    all_reward_strings = []
    reward_strings = ["# Reward function (s,a -> r)"]
    for expert in range(len(goodRocks)):
        for si in range(N):
            for sj in range(N):
                if (si,sj) in rockPositions.values():
                    k = [key for key, value in rockPositions.items() if value == (si,sj)][0]
                    for gr in range(pow(2,G)):
                        rew = "10" if status_gr(goodRocks[expert],k,gr,G) else "-10"
                        reward_strings.append(f"{state(si,sj,gr,N,G)},4 -> {rew}")
                if sj == N-1:
                    for gr in range(pow(2,G)):
                        reward_strings.append(f"{state(si,sj,gr,N,G)},2 -> 10")
        all_reward_strings.append("\n".join(reward_strings)+"\n\n")
        reward_strings = ["# Reward function (s,a -> r)"]
    
    belief_strings = ["# Initial beliefs (s -> p)"]
    belief_strings.append(f"{state(0,0,pow(2,G)-1,N,G)} -> 1")

    par_strings = [f"{0}, []"]

    for expert in range(len(goodRocks)):
        with open(filename+f"_expert{expert}.txt", "w") as file:
            file.write(state_string)
            file.write(env_string)
            file.write(action_string)
            file.write(obs_string)
            file.write("\n".join(par_strings)+"\n\n")
            file.write(all_transition_strings[expert])
            file.write(all_observation_strings[expert])
            file.write(all_reward_strings[expert])
            file.write("\n".join(belief_strings))

# Generate a POMDP for each environment in the ME-POMDP for the RockSample problem with an n x n grid, g good rocks, and k rocks in total, using random seed r to create the rock positions.
if __name__ == "__main__":
    n,g,k,r = 3,2,3,4
    Generate(f"../Models/RockSample_MEPOMDP_N{n}_G{g}_K{k}_R{r}",n,g,k,r)