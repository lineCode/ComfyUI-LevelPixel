In this node pack you will find:
- Model Unloader - a node that automatically unloads all checkpoints from memory.
It must be added to a sequential chain of nodes in the workflow. 
There are three versions of this node: 
Hard (complete unloading of all checkpoints from memory, except for GGUF (not supported yet)), 
Middle (the same as Hard, but in the future I plan to add widgets with the ability to select a mode), 
Soft (without unloading checkpoints from memory, just soft cleaning of memory from garbage).

- LLM Optional Memory Free Advanced - A node that generates text using the LLM model with subsequent unloading of the model from memory. 
Useful in those workflows where there is constant switching between different models and technologies under conditions of insufficient RAM of the video processor.
