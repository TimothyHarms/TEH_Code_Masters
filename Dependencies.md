# Dependencies used in Workflow
## ----Software----
1. Perl 5.12.3
2. Python 3+
3. ANSYS MAPDL 
4. Paraview (results viewer)
## ----Python----
1. numpy
2. scipy
3. seaborn
4. Pandas
5. hnsw
## Changes to code based on system
Edit system environment variables path and PYTHONPATH to include the ANSYS exe repository: `C:\Program Files\ANSYS Inc\ANSYS Student\vXXX\ANSYS\bin\winx64`
Edit `.\Project_Gastro\libraries\RemodellingRoutines\execute_ansys.pm` to reflect differing `ansysXXX.exe`
## Conda package info
If you wish to use these packages in a conda environment:
use code `conda list -e > pythonenvs.txt` to update conda environment packages
use code `conda install --file pythonenvs.txt` within your environment to install the required packages