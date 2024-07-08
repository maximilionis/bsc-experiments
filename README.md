# Bachelor Thesis Project

This project contains code developed for my bachelor thesis.

## Usage

- `main.py`: Processes and experiments with sets of three consecutive scenes.
- `main_setsoftwo.py`: Processes sets of two consecutive scenes.
- `Dashboard.py`: Used to view a dashboard for manually labeling data.
- `locations.csv`: Used for filtering and naming data variables, as well as for regridding the scenes.
- `requirements.txt`: Should allow for easy installation of needed libraries. 

## Important Notes

- If you want to load labeled data, please adjust the code. Currently, the code assigns labels based on location IDs (seen in `locations.csv`).
- The repository has only been used and tested on the Sentinel 2P TROPOMI data gathered in 2021.
- The `xesmf` library might not work without defining the folder where it has been installed. This is a bug know to them, and the solution is shown in `main.py` or `main_setsoftwo.py` to still get `xesmf` to work.