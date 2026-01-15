# Conda Commands: #
* Creating the environment & installing all dependencies (do this once at the start):
```conda env create -f python/environment.yml```
* Activating/entering the environment:
```conda activate capstone```
* Exiting the environment:
```conda deactivate```
* Updating the environment (if you add more dependencies):
```conda env update --file python/environment.yml --prune```
* Updating the environment file to match the current environment (please avoid this, apparently it can mess up the environment file):
```conda env export > python/environment_updated.yml```