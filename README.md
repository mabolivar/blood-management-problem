# A Blood Management problem - Simulator + Policy evaluator

This repository was developed to test adn compare different solution approaches (policies) for solving a 
Dynamic and Stochastic Inventory Management problem in a Blood Management context. 
It includes a Simulator based on Monte Carlo sampling and a set of different policies available for comparisson.
For more details about the problem details and policies, please refer to
[How to solve Dynamic Resource Allocation problems using RL](https://www.manuelbolivar.com/post/dynamic-resource-allocation/)
blog post.

This project was inspired by Powell's Approximate Dynamic Programing book []()
and the [wbpowell328/stochastic-optimization](https://github.com/wbpowell328/stochastic-optimization) repository.

## Setup

This project uses `Python 3.9` and all required packages can be found at the `requirements.txt` file.

## Usage

[blood-management-driver.py](blood-management-driver.py) is the starting point of the project and drives the execution
the simulation execution. The following params can use to trigger the execution:

```
"policies": ["myopic", "adp"], # Available policies: "myopic", "adp", and "basic"
"test_seed": 7383,             # Random seed for reproducibility
"test_simulations": 100,        # Number of simulations to run
"verbose": True,                # Prints all scenarios' performance (reward and gap to perfect solution)
"scenarios_to_visualize": 0,    # Number of scenarios to be graph at the `figures` folder
```

## Resources

+ https://www.manuelbolivar.com/post/dynamic-resource-allocation/
