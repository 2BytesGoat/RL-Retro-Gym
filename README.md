# Useful info
## Create local environment
`python -m virtualenv .env` <- *creates default environment* \
`.env\Scripts\activate` <- *activate environment* \
`pip install -r requirements.txt` <-- *install requirements*
`pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html` <-- *install pytorch*

## List supported roms
`from gym import envs; print(envs.registry.all())` <- *Python code for gym* \
`import retro; retro.data.list_games()` <- *Python code for gym retro* 

## Loading local roms
`python -m retro.import .\<path_to_roms_folder>\`

## How to get roms
[Old Atari Games - AtariMania](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) <- *some free Atari games*\
[Sega Genesis Classics - Steam](https://store.steampowered.com/agecheck/app/34270/) <- *buy games and load them into retro* \
[Ripping Roms from Cartriges - RetroGameBoards](https://www.retrogameboards.com/t/the-ripping-thread-how-to-build-your-own-legit-retro-rom-library/98) <- *how to extract games from physical cartriges*

## Add custom roms
### *Note to tinkers*: 

Games like **SuperMarioKart** are not supported by retro gym, but you can make them work yourself. All you need is the roms and to follow up the instructions bellow.\
I've also provided my own files which should work if you allready have a *rom.sfc* file.

### *How to use my files*:

**Step 1:** Open *.env\Lib\site-packages\retro\data\stable* in a new window (this we call the **"stable folder"**)\
**Step 2:** Place your *rom.sfc* file into *environments\SuperMarioKart-Snes*\
**Step 3:** Copy *SuperMarioKart-Snes* into the **stable folder**\
**Step 4:** `import retro; env=retro.make(game='SuperMarioKart-Snes');`

## Troubleshooting
[OpenAI Suported Games - Github](https://github.com/openai/retro/issues/53) <- *importing allready supported games in retro*\
[Integrating Games in Retro - Retro.io](https://retro.readthedocs.io/en/latest/integration.html) <- *official documentation for integrating new games in retro*\
[Game Integration UI - Retro.io](https://retro.readthedocs.io/en/latest/integration.html#the-integration-ui) <- *where to download the integration ui*\
[Integration UI issues - Retro.io](https://github.com/openai/retro/issues/159) <- *problems using the integration ui*

# TODOs
* Add support for CPU training
* Create models folder and add mlp encoders
* Remove OpenCV dependencies

# References

[Lucas Thompson - Youtube](https://www.youtube.com/channel/UCLA_tAh0hX9bjl6DfCe9OLw) <- retro-gym integration and general good starting point

