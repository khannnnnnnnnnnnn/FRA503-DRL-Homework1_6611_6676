FRA503 Deep Reinforcement Learning for Robotics
Instruction
Recommend using Miniconda
Download Miniconda different version, IsaacLab using python version 3.10 [list of Miniconda].

curl -O https://repo.anaconda.com/miniconda/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh
Install Miniconda

bash ~/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh
The installer finishes and displays, “Thank you for installing Miniconda3!”

Close and re-open your terminal window for the installation to fully take effect, or use the following command to refresh the terminal

source ~/.bashrc
Verifying the Miniconda installation
Test your installation by running conda list. If conda has been installed correctly, a list of installed packages appears.

image-1

If you see this, then the installation was successful! 🎉

Installing Isaac Sim & Isaac Lab
Pip Installation (recommended for Ubuntu 22.04)
Follow the Installing and Verifying steps [link]

⚠️ Important Notice

IsaacLab must be installed from the release/2.1.0 branch to ensure compatibility. Installing from other branches may lead to errors or unexpected behavior.

git clone -b release/2.1.0 https://github.com/isaac-sim/IsaacLab.git
Verifying the Isaac Lab installation
# Option 1: Using the isaaclab.sh executable
# note: this works for both the bundled python and the virtual environment
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

# Option 2: Using python in your virtual environment
python scripts/tutorials/00_sim/create_empty.py
image

If you see this, then the installation was successful! 🎉

Isaac Lab Overview
This overview introduces key concepts in IsaacLab. Focus on the [link] required for this class.

Optional sections provide deeper understanding - read these or explore the full IsaacLab documentation based on your interests.

After reading each section, you should be able to answer these guiding questions:

Core Concepts

1.1 Task Design Workflows [link]

What is the different between Manager-based and Direct workflows?

If you're just starting to use IsaacLab, which workflow should you choose?

1.2 Actuators [Optional] [link]

How does the physics engine handle position and velocity control differently from torque control?

What limitation exists when simulating actuators compared to real-world robot behavior?

Developer’s Guide

2.1 Setting up Visual Studio Code [link]

How to setup VS Code IDE for debugging python code?

How to use different python interpreters?

2.2 Repository organization [link]

How to find source code for all Isaac Lab extensions and standalone applications?

What is the different between extensions and standalone applications?

2.3 Application Development [Optional] [link]

Why do scripts need to be structured as extensions and standalone applications?
2.4 Building your Own Project [Optional] [link]

Sensors [Optional]

This section provides an overview of the sensor APIs available in Isaac Lab.

3.1 Camera [link]

3.2 Contact Sensor [link]

3.3 Frame Transformer [link]

3.4 Ray Caster [link]

Available Environments
The following lists comprises of all the RL tasks implementations that are available in Isaac Lab. [link]

or

you can excute following command line

python scripts/environments/list_envs.py
image-2

Tutorials
We recommend that you go through the tutorials in the order they are listed here.

Note: There is no need to follow the order of the tutorial in page via next button.

Simulation Overview
Setting up a Simple Simulation

1.1 Creating an empty scene [link]

1.2 Spawning prims into the scene [link]

1.3 Deep-dive into AppLauncher [link]

Interacting with Assets

2.1 Interacting with a rigid object [link]

2.2 Interacting with an articulation [link]

Creating a Scene

3.1 Using the Interactive Scene [link]

Task Design Workflows
For more detail of different workflows for designing environments. [link]

Designing an Environment [link]

HW0 Requirement: You need to understand Creating a Manager-Based Base Environment and Creating a Manager-Based RL Environment for designing an environment.

4.1 Creating a Manager-Based Base Environment [link]

4.2 Creating a Manager-Based RL Environment [link]

4.3 Registering an Environment [link]

4.4 Training with an RL Agent [link]

How-to Guides [Optional]
This section includes guides that help you use Isaac Lab. These are intended for users who have already worked through the tutorials and are looking for more information on how to use Isaac Lab. If you are new to Isaac Lab, we recommend you start with the tutorials. [link]
