
import numpy as np
import torch as th
import os
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


plt.style.use("dark_background")


def determine_device():
    """
    This function evaluates whether a GPU is accessible at the system and
    returns it as device to calculate on, otherwise it returns the CPU.
    :return: The device where tensor calculations shall be made on
    """
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    # Additional Info when using cuda
    if device.type == "cuda":
        print(th.cuda.get_device_name(0))
        print("Memory Usage:")
        print("\tAllocated:",
              round(th.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("\tCached:   ", round(th.cuda.memory_reserved(0) / 1024 ** 3, 1),
              "GB")
        print()
    return device


def save_model_to_file(path, cfg_file, current_epoch, epochs, epoch_errors,
                       net, model_name):
    
    # Create the directory to save the model
    os.makedirs(path, exist_ok=True)

    # Save model weights to file
    th.save(net.state_dict(), path + "/" + model_name + ".pt")

    output_string = cfg_file + "\n#\n# Performance\n\n"

    output_string += "CURRENT_EPOCH = " + str(current_epoch) + "\n"
    output_string += "EPOCHS = " + str(epochs) + "\n"
    output_string += "CURRENT_TRAINING_ERROR = " + \
                     str(epoch_errors[-1]) + "\n"
    output_string += "LOWEST_TRAINING_ERROR = " + \
                     str(min(epoch_errors)) + "\n"

    # Save the configuration and current performance to file
    with open(path + "cfg_and_performance.txt", "w") as text_file:
        text_file.write(output_string)


def animate_diffusion(outputs_dis, outputs_tot, targets_dis, targets_tot,
                      teacher_forcing_steps):
    
    # First set up the figure, the axis, and the plot element we want to
    # animate
    fig, axes = plt.subplots(2, 1, figsize=[12, 6], dpi=100)
    axes[0].set_ylim(min(np.min(outputs_dis), np.min(targets_dis)),
                     max(np.max(outputs_dis), np.max(targets_dis)))
    axes[0].set_ylabel("Dissolved concentration")
    axes[1].set_ylim(min(np.min(outputs_tot), np.min(targets_tot)),
                     max(np.max(outputs_tot), np.max(targets_tot)))
    axes[1].set_ylabel("Total")

    txt = axes[0].text(0, axes[0].get_yticks()[-1], "t = 0", fontsize=20,
                       color="white")

    # Plot the dissorption
    outputs_dis_lines = []
    for n in range(len(outputs_dis)):
        if n == 0:
            label="Network output"
        else:
            label=None
        diss_line, = axes[0].plot(
            range(len(outputs_dis[0])), outputs_dis[n, :, 0],
            label=label, color="deepskyblue"
        )
        outputs_dis_lines.append(diss_line)

    targets_dis_line, = axes[0].plot(
        range(len(targets_dis[0])), targets_dis[0, :, 0], label="Ground truth",
        color="red", linestyle="--"
    )

    axes[0].legend()

    # Plot the total amount
    outputs_tot_lines = []
    for n in range(len(outputs_tot)):
        tot_line, = axes[1].plot(
            range(len(outputs_tot[0])), outputs_tot[n, :, 0],
            color="deepskyblue"
        )
        outputs_tot_lines.append(tot_line)

    targets_tot_line, = axes[1].plot(
        range(len(targets_tot[0])), targets_tot[0, :, 0], color="red",
        linestyle="--"
    )

    anim = animation.FuncAnimation(fig, animate, frames=len(outputs_dis[0, 0]),
                                   fargs=(outputs_dis, outputs_tot,
                                          targets_dis, targets_tot,
                                          outputs_dis_lines, outputs_tot_lines,
                                          targets_dis_line, targets_tot_line,
                                          txt, teacher_forcing_steps),
                                   interval=1)

    plt.show()

    return anim


def animate(i, outputs_dis, outputs_tot, targets_dis, targets_tot,
            outputs_dis_lines, outputs_tot_lines, targets_dis_line,
            targets_tot_line, txt, teacher_forcing_steps):

    # Pause the simulation briefly when switching from teacher forcing to
    # closed loop prediction
    if i == teacher_forcing_steps:
        time.sleep(1.0)

    # Display the current timestep in text form in the plot
    if i < teacher_forcing_steps:
        txt.set_text("Teacher forcing, t = " + str(i))
    else:
        txt.set_text("Closed loop prediction, t = " + str(i))

    # Update the dissolve plot
    for line_idx, outputs_dis_line in enumerate(outputs_dis_lines):
        outputs_dis_line.set_ydata(outputs_dis[line_idx, :, i])
    targets_dis_line.set_ydata(targets_dis[0, :, i])

    # Update the total plot
    for line_idx, outputs_tot_line in enumerate(outputs_tot_lines):
        outputs_tot_line.set_ydata(outputs_tot[line_idx, :, i])
    targets_tot_line.set_ydata(targets_tot[0, :, i])

    return outputs_dis_lines, outputs_tot_lines, \
           targets_dis_line, targets_tot_line
