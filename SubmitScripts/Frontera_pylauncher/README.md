The files in this directory use the `pylauncher` module on TACC's Frontera to run multiple Python3 commands on a single node in order to improve resource usage efficiency.

1. List all the commands you want to run in the file `pylauncher_tasks.txt`
   **NOTE:** limit the number of commands according to the amount of memory used by the scripts in your use case and the total available memory in the node (~188GB)
2. Edit the job submission script if needed (e.g. partition, walltime, ...)
3. Submit the job by running
   ```
   sbatch Frontera_pylauncher.sub
   ```
