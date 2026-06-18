# EMG Robot Dog

EMG-based control system for the Freenove Robot Dog Kit (FNK0050), reusing the robot's original communication protocol.

## Overview

This project adds an EMG-based control layer on top of the manufacturer's original Freenove Robot Dog firmware and client-server architecture, allowing the robot to be controlled in real time through EMG signals instead of the default interface.

## Team

| Name | Contact |
| --- | --- |
| Afonso Sousa | up202207498@up.pt |
| Maria Matos | up202208005@up.pt |
| Sara Gouveia | up202206979@up.pt |

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
