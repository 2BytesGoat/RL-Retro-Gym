# Setup

## To get the list of the supported games run

`from gym import envs; 
print(envs.registry.all())` - *for gym*

`import retro; 
retro.data.list_games()` - *for gym retro*

## To add a game which is *supported* by retro run

`python -m retro.import .\<path_to_roms_folder>\`

## To obtain roms, you can purchase some of the games on 
[Steam](https://store.steampowered.com/agecheck/app/34270/) - Sega Genesis Classics

## A list of *custom* environemnt data without roms is located at

*.\environments*

## Important!
In order for retro to see your custom rom you must move it to **retro\data\stable**

# References

[Lucas Thompson](https://www.youtube.com/channel/UCLA_tAh0hX9bjl6DfCe9OLw) - 
retro-gym integration and general good starting point

# Troubleshooting
Importing allready supported games in retro go 
[here](https://github.com/openai/retro/issues/53)

Official documentation for integrating new games in retro go
[here](https://retro.readthedocs.io/en/latest/integration.html)

Cant find where to download the integration ui go 
[here](https://retro.readthedocs.io/en/latest/integration.html#the-integration-ui)

Having problems using the integration ui go
[here](https://github.com/openai/retro/issues/159)

Legitimate procedure for extracting rom files
[here](https://www.retrogameboards.com/t/the-ripping-thread-how-to-build-your-own-legit-retro-rom-library/98)
