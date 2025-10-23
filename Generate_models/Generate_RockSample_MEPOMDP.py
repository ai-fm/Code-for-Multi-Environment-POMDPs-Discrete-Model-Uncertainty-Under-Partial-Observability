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
    
    envs = []
    for e in range(len(goodRocks)):
        env = ""
        for k in range(K):
            if k in goodRocks[e]:
                env += "g"
            else:
                env += "b"
        envs.append(env)
    envs = ", ".join([f"e-{env}" for env in envs])
    env_string = f"{len(goodRocks)}, [{envs}]\n"
    
    A = K+5
    check_actions = ", ".join([f"Check{k}" for k in range(K)])
    action_string = f"{A}, [North, South, East, West, Sample, {check_actions}]\n"
    obs_string = "3, [Bad, Good, Nothing]\n\n"

    transition_strings = ["# Transition function (s,a,s -> p)"]
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
                    rest_prob = "1"
                    gr_bin = format(gr, f'0{G}b')
                    for g in range(G):
                        if gr_bin[g] == '1':
                            gr_bin = gr_bin[0:g]+"0"+gr_bin[g+1:]
                            gr2 = int(gr_bin,2)
                            pr = f"pt_{k}.{g}"
                            transition_strings.append(f"{state(si,sj,gr,N,G)},4,{state(si,sj,gr2,N,G)} -> {pr}")
                            rest_prob += f"-{pr}"
                    transition_strings.append(f"{state(si,sj,gr,N,G)},4,{state(si,sj,gr,N,G)} -> {rest_prob}")
            for k in range(K):
                a = k+5
                for gr in range(pow(2,G)):
                    transition_strings.append(f"{state(si,sj,gr,N,G)},{a},{state(si,sj,gr,N,G)} -> 1")
    transition_strings.append(f"{S-1},_,{S-1} -> 1")

    half_dist = math.floor(N/2)
    eff_lev = dict()
    observation_strings = ["# Observation function (a,s,o -> p)"]
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
                    observation_strings.append(f"{a},{state(si,sj,gr,N,G)},1 -> po_{k}.{gr_bin}_{c_eff}")
                    observation_strings.append(f"{a},{state(si,sj,gr,N,G)},0 -> 1-po_{k}.{gr_bin}_{c_eff}")
    observation_strings.append(f"_,{S-1},2 -> 1")

    reward_strings = ["# Reward function (s,a -> r)"]
    for si in range(N):
        for sj in range(N):
            if (si,sj) in rockPositions.values():
                k = [key for key, value in rockPositions.items() if value == (si,sj)][0]
                for gr in range(pow(2,G)):
                    gr_bin = format(gr, f'0{G}b')
                    reward_strings.append(f"{state(si,sj,gr,N,G)},4 -> r_{k}.{gr_bin}")
            if sj == N-1:
                for gr in range(pow(2,G)):
                    reward_strings.append(f"{state(si,sj,gr,N,G)},2 -> 10")
    
    belief_strings = ["# Initial beliefs (s -> p)"]
    belief_strings.append(f"{state(0,0,pow(2,G)-1,N,G)} -> 1")

    par_list = [f"pt_{k}.{g}" for k in range(K) for g in range(G)]
    par_list += [f"po_{k}.{format(gr, f'0{G}b')}_{c_eff}" for k in range(K) for gr in range(pow(2,G)) for c_eff in range(len(eff_lev))]
    par_list += [f"r_{k}.{format(gr, f'0{G}b')}" for k in range(K) for gr in range(pow(2,G))]
    pars = ", ".join(par_list)
    par_strings = [f"{len(par_list)}, [{pars}]"]
    for e in range(len(goodRocks)):
        transition_pars = ["1" if status_g(goodRocks[e],k,g) else "0" for k in range(K) for g in range(G)]
        transition_pars = ", ".join(transition_pars)

        observation_pars = [f"{round(0.5 + 0.5*eff_lev[c_eff],6)}" if status_gr(goodRocks[e],k,gr,G) else f"{round(1 - (0.5 + 0.5*eff_lev[c_eff]),6)}" for k in range(K) for gr in range(pow(2,G)) for c_eff in range(len(eff_lev))]
        observation_pars = ", ".join(observation_pars)
        
        reward_pars = ["10" if status_gr(goodRocks[e],k,gr,G) else "-10" for k in range(K) for gr in range(pow(2,G))]
        reward_pars = ", ".join(reward_pars)
        par_strings.append(f"[{transition_pars}, {observation_pars}, {reward_pars}]")
    
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

# Generate a ME-POMDP for the RockSample problem with an n x n grid, g good rocks, and k rocks in total, using random seed r to create the rock positions.
if __name__ == "__main__":
    n,g,k,r = 3,2,4,72
    Generate(f"Models/RockSample_MEPOMDP_N{n}_G{g}_K{k}_R{r}.txt",n,g,k,r)