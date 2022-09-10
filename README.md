# Mario Kart SNES Agent
 
Dabble with computer vision and machine learning by training a Mario Kart SNES AI. Use this repository to make your process smoother.
 
<center><img src="./docs/ezgif-1-d0f24e7559.gif"/></center>
 
## Experiment ideas
- [ ] Add information about kart speed
- [ ] Use Convolutional Networks to embed the state
- [ ] Train an agent using the [NEAT algorithm](https://neat-python.readthedocs.io/en/latest/)
- [ ] Train a [DQN](https://stable-baselines.readthedocs.io/en/master/modules/dqn.html) agent using reinforcement learning
 
## How to setup
 
### **Step 0 - Downloading prerequisites**
I don't condone the usage of Anaconda because it seemed bloated, but sometimes it's useful because you can easily install a specific python version inside your virtual environment.
 
You can download Anaconda for your OS of choice from [this link](https://www.anaconda.com/products/distribution).
 
If you can't call `conda` from your shell of choice, you may have to [manually add it to path](https://www.geeksforgeeks.org/how-to-setup-anaconda-path-to-environment-variable).
 
Also, you may need to install FFMPEG to be able to save recordings of your own gameplay. A windows tutorial of how you can do this, can be downloaded from [here](https://ffmpeg.org/download.html#build-windows) and a setup tutorial can be found [here](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/).
 
### **Step 1 - Create a conda environment**
 
```
conda create --name rl-playground python=3.7
conda activate rl-playground
pip install -r requirements.txt
```
As of writing this doc retro-gym has max support for Python 3.7. \
Check [here](https://retro.readthedocs.io/en/latest/getting_started.html) for the latest version.
 
### **Step 2 - Copy the ROM to retro-gym's **stable** game folder**
 
**Note**: you will need to find your own `rom.sfc` and `rom.sha` files because Nintendo does not approve sharing of their ROMs and does not have any way of legally acquiring them.
 
There are some **grayarea** methods to obtain ROMs ([like the ones from here](https://arekuse.net/blog/tech-guides/rom-dumping-and-hacking/rom-dumping-and-flashing-genesis-mega-drive/)) but unfortunately I cannot share them.
 
Due to the nature of the ROM, OpenAI does not support it by default.
```
#Linux
/home/user/.conda/envs/rl-playground/lib/python3.7/site-packages/retro/data/stable
 
#Windows
C:/Users/.conda/Miniconda3/envs/rl-playground/Lib/site-packages/retro/data/stable
```
 
Alternatively, you can run `conda env list` and see where your conda environment is located.
 
### **Step 3 - Test that everything is running**
 
Run the random agent provided
```
python 01_random_agent.py
```
 
## Using the Retro Integration UI
 
### **Step 1 - Move the exe to the folder which contains retro**
 
You have to run `import retro; print(retro.__path__)`! It'll display a file path and you have to move the .exe file to there and run it.
 
### **Step 2 - Integrate the ROM**
 
Execute `Ctrl + Shift + O` and locate the ROM in your filesystem.
 
## Downloading gameplay for Behavioral Cloning
 
The data that I have used to train the model can be found [on this Google Drive](https://drive.google.com/drive/folders/13e1j-iej0eMHftiAwfoiBw4g560dTglT?usp=sharing).
 
## Special thanks to
 
* Esteveste - **Started code for Mario Kart** - [GitHub](https://github.com/esteveste/gym-SuperMarioKart-Snes).
 
## Known issues
 
* If `00_manual_kart.py` is missing or getting deleted by Windows defender (because it contains a key listener), check the following [link](https://answers.microsoft.com/en-us/protect/forum/all/windows-defender-deleting-my-own-applications/65b96548-e411-42b0-8ce1-ae28f46eb9a1) to setup exceptions for this folder.
 