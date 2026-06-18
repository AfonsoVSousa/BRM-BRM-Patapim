# EMG Robot Dog

EMG-based control system for the Freenove Robot Dog Kit (FNK0050), reusing the robot's original communication protocol.

## Overview

This project adds an EMG-based control layer on top of the manufacturer's original Freenove Robot Dog firmware and client-server architecture, allowing the robot to be controlled in real time through EMG signals instead of the default interface.

## Team

| Name | Contact |
| --- | --- |
| Afonso Sousa | up202207498@up.pt |
| André Pereira | up202206403@up.pt |
| Maria Matos | up202208005@up.pt |
| Sara Gouveia | up202206979@up.pt |

## Project Structure

```text
BRM-BRM-Patapim/
├── ourmain.py           # Entry point that launches the PyQt GUI
├── ui.py                # Main interface, plots, calibration, DAQ, robot orchestration and runtime control
├── emg_core.py          # EMG signal generation, filtering, normalization, and calibration logic
├── config.py            # Shared constants, thresholds, filter coefficients, and hardware settings
├── robot_controller.py  # Freenove robot client wrapper and EMG → command mapping
├── frontalis_daq.py     # NI-DAQmx frontalis output/input loop for hardware signal injection
└── README.md            # Project overview and usage notes
```

## Instructions

1. **Download the original robot code**: [Freenove_Robot_Dog_Kit_for_Raspberry_Pi](https://github.com/Freenove/Freenove_Robot_Dog_Kit_for_Raspberry_Pi)

2. **Download our project repository** and place its files inside the `Client` folder of the original code, alongside the existing files.

3. **Connect to the robot via VNC**:
   - Turn on the **S1** and **S2** switches on the robot
   - Find the robot's IP address (use an IP Scanner app if unknown)
   - Open VNC Viewer and connect using that IP address
   - A pop-up should appear with the Raspberry Pi desktop — open a terminal there
   - Run the server:
     ```bash
     cd ~/Freenove_Robot_Dog_Kit_for_Raspberry_Pi/Code/Server
     sudo python main.py
     ```

4. **Open a terminal on your own computer** (outside the robot's remote environment).

5. **Navigate to the `Client` folder**, inside the location where the code was saved.

6. **Run the EMG control GUI**:
   ```bash
   python ourmain.py
   ```

## Configuration

Before using the code, update the robot IP address and the NI-DAQmx device/channel names in `config.py` so they match your setup. It is also worth checking the remaining values in that file, such as the sampling rate, filter cutoffs, thresholds, and DAQ voltage limits, to make sure they are appropriate for your application and hardware.
