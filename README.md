# Irregular-Object-Packing
Planning Irregular Object Packing via Hierarchical Reinforcement Learning
This repo contains the basic code for Irregular Object Packing on PyBullet. 
This is an old version of our code and we are keeping updating.

## Quick Start

The code is based on PyTorch. Please prepare the object file from YCB and OCRTOC and put them into objects folder. Run env.generate() to generate index of object files. And then run main.py to train the model. 

 
## Repo organization 

The repo is organized as follows:
-	main.py: The main 
-	env.py: Functions to interact with the PyBullet simulation environment
-   models.py: Models for sequence planning and placement planning
-   trainer.py: Trainer for models
-   heuristics_HM.py: HM heuristic for placement
-   evaluate.py: Evaluate performance of different methods on C, P and S.

- Link for downloading objects:
  - YCB:
	- https://github.com/eleramp/pybullet-object-models
  - OCRTOC:
    - http://www.ocrtoc.org
