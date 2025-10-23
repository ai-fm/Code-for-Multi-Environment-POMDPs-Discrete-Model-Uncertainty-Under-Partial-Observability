# set matplotlib fonts to times-new-roman
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
matplotlib.font_manager.fontManager.addfont("Times New Roman.ttf")
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 19
import re
import numpy as np


def figure_rock_positions(saveLocation = None, axis_labels = True, files = None, sizes=None):
    data = []

    for i in range(len(files)):
        data_point = []
        with open(f"Results/{files[i][0]}", "r") as f:
            lines = f.readlines()
            line = re.split("\t|\n",[l for l in lines if not l.split("\t")[0].isdigit()][1])
            data_point.append(float(line[1]))
        with open(f"Results/{files[i][1]}", "r") as f:
            lines = f.readlines()
            line = re.split("\t|\n",[l for l in lines if not l.split("\t")[0].isdigit()][1])
            data_point.append(float(line[1]))
        data.append(data_point)


    labels = [f"RS$^\mathdefault{{c}}_\mathdefault{{{inf[0]},{inf[1]},{inf[2]}}}$" for inf in sizes]
    nearby_vals = [d[0] for d in data]
    faraway_vals = [d[1] for d in data]

    x = np.arange(len(labels))
    width = 0.35 

    fig, ax = plt.subplots(figsize=(8, 3))
    rects1 = ax.bar(x - width/2, nearby_vals, width, label='Nearby', edgecolor='black', facecolor='#08E8DE')
    rects2 = ax.bar(x + width/2, faraway_vals, width, label='Far away', edgecolor='black', facecolor='#FFAE42')

    if axis_labels:
        ax.set_ylabel('Convergence time (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend() 

    if saveLocation is not None:
        plt.savefig(saveLocation / "Rocks_nearby_vs_far_away.pdf", format="pdf", bbox_inches='tight')
        plt.savefig(saveLocation / "Rocks_nearby_vs_far_away.png", bbox_inches='tight')
    else:
        plt.show()


def figure_model_type(saveLocation = None, axis_labels = True, files = None, sizes=None):
    data = []

    for i in range(len(files)):
        data_point = []
        with open(f"Results/{files[i][0]}", "r") as f:
            lines = f.readlines()
            line = re.split("\t|\n",[l for l in lines if not l.split("\t")[0].isdigit()][1])
            data_point.append(float(line[1]))
        with open(f"Results/{files[i][1]}", "r") as f:
            lines = f.readlines()
            line = re.split("\t|\n",[l for l in lines if not l.split("\t")[0].isdigit()][1])
            data_point.append(float(line[1]))
        data.append(data_point)

    labels = [f"RS$_\mathdefault{{{inf[0]},{inf[1]},{inf[2]}}}$" for inf in sizes]
    abpomdp_vals = [d[0] for d in data]
    mepomdp_vals = [d[1] for d in data]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 3))
    rects1 = ax.bar(x - width/2, abpomdp_vals, width, label='AB-POMDP', edgecolor='black', facecolor='#08E8DE')
    rects2 = ax.bar(x + width/2, mepomdp_vals, width, label='ME-POMDP', edgecolor='black', facecolor='#FFAE42')

    if axis_labels:
        ax.set_ylabel('Convergence time (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    if saveLocation is not None:
        plt.savefig(saveLocation / "AB-POMDP_vs_ME-POMDP.pdf", format="pdf", bbox_inches='tight')
        plt.savefig(saveLocation / "AB-POMDP_vs_ME-POMDP.png", bbox_inches='tight')
    else:
        plt.show()


def figure_robustness(plot_me_pomdp = True, saveLocation = None, axis_labels = True, files = None, sizes=None):
    data = []

    for i in range(len(files)):
        data_point = []
        mepomdp_time = 0
        sum_pomdp_times = 0
        with open(f"Results/{files[i][0]}", "r") as f:
            lines = f.readlines()
            line = re.split("\t|\n",[l for l in lines if not l.split("\t")[0].isdigit()][1])
            data_point.append(float(line[0]))
            mepomdp_time = float(line[1])
        with open(f"Results/{files[i][1]}", "r") as f:
            lines = f.readlines()
            vals = [float(l.split()[-1]) for l in lines]
            data_point.append(max(vals))
            data_point.append(min(vals))
        for j in range(len(files[i][2:])):
            with open(f"Results/{files[i][j+2]}", "r") as f:
                lines = f.readlines()
                line = re.split("\t|\n",[l for l in lines if not l.split("\t")[0].isdigit()][1])
                data_point.append(float(line[0]))
                sum_pomdp_times += float(line[1])
        data_point.append(mepomdp_time/sum_pomdp_times)
        data.append(data_point)


    labels = [f"RS$_\mathdefault{{{inf[0]},{inf[1]},{inf[2]}}}$" for inf in sizes]
    pomdp_vals = [d[3:-1] for d in data]
    mepomdp_vals = [d[0] for d in data]
    lower_vals = [d[2] for d in data]

    pomdp_vals_concat = []

    x_pomdp = []
    x_mepomdp = []
    x_lower_vals = []
    x_labels = []

    width = 0.2
    bar_width_pomdp = 0.05
    bar_width = 0.2

    sub_group_offset = 0.0
    group_offset = 0.1

    x_curr = 0
    for d in pomdp_vals:
        x_sum = 0

        for v in d:
            pomdp_vals_concat.append(v)
            x_pomdp.append(x_curr)
            x_sum += x_curr
            x_curr += bar_width_pomdp

        x_curr += 0.5*(bar_width - bar_width_pomdp)
        x_curr += sub_group_offset

        x_mepomdp.append(x_curr)
        x_sum += x_curr
        x_curr += bar_width + sub_group_offset

        x_lower_vals.append(x_curr)
        x_sum += x_curr + 0.1
        x_curr += bar_width + group_offset

        x_labels.append(x_sum / (len(d) + 2))

    if not (len(x_mepomdp) == len(mepomdp_vals) == len(x_lower_vals) == len(lower_vals) == len(labels)):
        raise ValueError("Length mismatch")

    fig, ax = plt.subplots(figsize=(12, 5.5))
    rects1 = ax.bar(x_pomdp, pomdp_vals_concat, bar_width_pomdp, label='Individual POMDPs', edgecolor='black', facecolor='#08E8DE')
    if plot_me_pomdp:
        rects2 = ax.bar(x_mepomdp, mepomdp_vals, width, label='ME-POMDP', edgecolor='black', facecolor='#FFAE42')
    rects3 = ax.bar(x_lower_vals, lower_vals, width, label='Worst-case incorrect POMDP', edgecolor='black', facecolor='#9ACD32')
    
    if axis_labels:
        ax.set_ylabel('Value (higher is better)')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x_labels)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(-5,30)

    if saveLocation is not None:
        if plot_me_pomdp:
            plt.savefig(saveLocation / "robustness.pdf", format="pdf", bbox_inches='tight')
            plt.savefig(saveLocation / "robustness.png", bbox_inches='tight')
        else:
            plt.savefig(saveLocation / "robustness_no_mepomdp.pdf", format="pdf", bbox_inches='tight')
            plt.savefig(saveLocation / "robustness_no_mepomdp.png", bbox_inches='tight')
    else:
        plt.show()


def figure_scaling_state_space(saveLocation = None, axis_labels = True, files = None, state_spaces=None):
    data = []

    for i in range(len(files)):
        sub_data = []
        for j in range(len(files[i])):
            with open(f"Results/{files[i][j]}", "r") as f:
                lines = f.readlines()
                line = re.split("\t|\n",[l for l in lines if not l.split("\t")[0].isdigit()][1])
                sub_data.append(float(line[1]))
        data.append(sub_data)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    rects1 = ax.plot(state_spaces[0],data[0],label="2 rocks, 1 good",color='#08E8DE', marker='o')
    rects2 = ax.plot(state_spaces[1],data[1],label="3 rocks, 1 good",color='#FFAE42', marker='^')
    rects2 = ax.plot(state_spaces[2],data[2],label="3 rocks, 2 good",color='#9ACD32', marker='D')
    ax.set_xlim(0,105)

    if axis_labels:
        ax.set_ylabel('Convergence time (s)')
        ax.set_xlabel('Number of states')
    ax.legend() 

    if saveLocation is not None:
        plt.savefig(saveLocation / "Rocks_scaling_state_space.pdf", format="pdf", bbox_inches='tight')
        plt.savefig(saveLocation / "Rocks_scaling_state_space.png", bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    from pathlib import Path

    saveLocation = Path("Figures")
    if not saveLocation.exists():
        saveLocation.mkdir(parents=True, exist_ok=True)

    axis_labels = True

    files1 = [["RockSample_MEPOMDP_2corners_N2_G1.txt","RockSample_MEPOMDP_2corners_N2_G1.txt"],
              ["RockSample_MEPOMDP_2corners_close_N3_G1.txt","RockSample_MEPOMDP_2corners_N3_G1.txt"],
              ["RockSample_MEPOMDP_2corners_close_N4_G1.txt","RockSample_MEPOMDP_2corners_N4_G1.txt"],
              ["RockSample_MEPOMDP_corners_N2_G1.txt","RockSample_MEPOMDP_corners_N2_G1.txt"],
              ["RockSample_MEPOMDP_corners_close_N3_G1.txt","RockSample_MEPOMDP_corners_N3_G1.txt"]
              ]
    sizes1 = [[2,1,2],[3,1,2],[4,1,2],[2,1,3],[3,1,3]]
    figure_rock_positions(saveLocation=saveLocation, axis_labels=axis_labels, files=files1, sizes=sizes1)

    files2 = [["RockSample_POMDP_N3_G1_K2_R18.txt","RockSample_MEPOMDP_N3_G1_K2_R18.txt"],
              ["RockSample_POMDP_N3_G1_K3_R13.txt","RockSample_MEPOMDP_N3_G1_K3_R13.txt"],
              ["RockSample_POMDP_N3_G1_K4_R83.txt","RockSample_MEPOMDP_N3_G1_K4_R83.txt"],
              ["RockSample_POMDP_N3_G2_K3_R4.txt","RockSample_MEPOMDP_N3_G2_K3_R4.txt"],
              ["RockSample_POMDP_N4_G1_K2_R33.txt","RockSample_MEPOMDP_N4_G1_K2_R33.txt"],
              ["RockSample_POMDP_N5_G1_K2_R96.txt","RockSample_MEPOMDP_N5_G1_K2_R96.txt"],
              ["RockSample_POMDP_N6_G1_K2_R77.txt","RockSample_MEPOMDP_N6_G1_K2_R77.txt"]
              ]
    sizes2 = [[3,1,2],[3,1,3],[3,1,4],[3,2,3],[4,1,2],[5,1,2],[6,1,2]]
    figure_model_type(saveLocation=saveLocation, axis_labels=axis_labels, files=files2, sizes=sizes2)

    files3 = [["RockSample_MEPOMDP_N3_G1_K2_R18.txt", "RockSample_MEPOMDP_N3_G1_K2_R18_expert_summary.txt", "RockSample_MEPOMDP_N3_G1_K2_R18_expert0.txt", "RockSample_MEPOMDP_N3_G1_K2_R18_expert1.txt"],
              ["RockSample_MEPOMDP_N3_G1_K3_R13.txt", "RockSample_MEPOMDP_N3_G1_K3_R13_expert_summary.txt", "RockSample_MEPOMDP_N3_G1_K3_R13_expert0.txt", "RockSample_MEPOMDP_N3_G1_K3_R13_expert1.txt", "RockSample_MEPOMDP_N3_G1_K3_R13_expert2.txt"],
              ["RockSample_MEPOMDP_N3_G1_K4_R83.txt", "RockSample_MEPOMDP_N3_G1_K4_R83_expert_summary.txt", "RockSample_MEPOMDP_N3_G1_K4_R83_expert0.txt", "RockSample_MEPOMDP_N3_G1_K4_R83_expert1.txt", "RockSample_MEPOMDP_N3_G1_K4_R83_expert2.txt", "RockSample_MEPOMDP_N3_G1_K4_R83_expert3.txt"],
              ["RockSample_MEPOMDP_N3_G2_K3_R4.txt", "RockSample_MEPOMDP_N3_G2_K3_R4_expert_summary.txt", "RockSample_MEPOMDP_N3_G2_K3_R4_expert0.txt", "RockSample_MEPOMDP_N3_G2_K3_R4_expert1.txt", "RockSample_MEPOMDP_N3_G2_K3_R4_expert2.txt"],
              ["RockSample_MEPOMDP_N4_G1_K2_R33.txt", "RockSample_MEPOMDP_N4_G1_K2_R33_expert_summary.txt", "RockSample_MEPOMDP_N4_G1_K2_R33_expert0.txt", "RockSample_MEPOMDP_N4_G1_K2_R33_expert1.txt"],
              ["RockSample_MEPOMDP_N5_G1_K2_R96.txt", "RockSample_MEPOMDP_N5_G1_K2_R96_expert_summary.txt", "RockSample_MEPOMDP_N5_G1_K2_R96_expert0.txt", "RockSample_MEPOMDP_N5_G1_K2_R96_expert1.txt"],
              ["RockSample_MEPOMDP_N6_G1_K2_R77.txt", "RockSample_MEPOMDP_N6_G1_K2_R77_expert_summary.txt", "RockSample_MEPOMDP_N6_G1_K2_R77_expert0.txt", "RockSample_MEPOMDP_N6_G1_K2_R77_expert1.txt"]
              ]
    sizes3 = [[3,1,2],[3,1,3],[3,1,4],[3,2,3],[4,1,2],[5,1,2],[6,1,2]]
    figure_robustness(plot_me_pomdp=True, saveLocation=saveLocation, axis_labels=axis_labels, files=files3, sizes=sizes3)

    files4 = [["RockSample_MEPOMDP_2corners_N2_G1.txt",
              "RockSample_MEPOMDP_2corners_close_N3_G1.txt",
              "RockSample_MEPOMDP_2corners_close_N4_G1.txt",
              "RockSample_MEPOMDP_2corners_close_N5_G1.txt",
              "RockSample_MEPOMDP_2corners_close_N6_G1.txt",
              "RockSample_MEPOMDP_2corners_close_N7_G1.txt"],
              ["RockSample_MEPOMDP_corners_N2_G1.txt",
              "RockSample_MEPOMDP_corners_close_N3_G1.txt",
              "RockSample_MEPOMDP_corners_close_N4_G1.txt",
              "RockSample_MEPOMDP_corners_close_N5_G1.txt",
              "RockSample_MEPOMDP_corners_close_N6_G1.txt",
              "RockSample_MEPOMDP_corners_close_N7_G1.txt"],
              ["RockSample_MEPOMDP_corners_N2_G2.txt",
              "RockSample_MEPOMDP_corners_close_N3_G2.txt",
              "RockSample_MEPOMDP_corners_close_N4_G2.txt",
              "RockSample_MEPOMDP_corners_close_N5_G2.txt"]
              ]
    state_spaces4 = [[9,19,33,51,73,99], [9,19,33,51,73,99], [17,37,65,101]]
    figure_scaling_state_space(saveLocation=saveLocation, axis_labels=axis_labels, files=files4, state_spaces=state_spaces4)
