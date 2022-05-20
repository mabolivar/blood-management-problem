# A Blood Management problem - Simulator + Policy evaluator

This repository was developed to test adn compare different solution approaches (policies) for solving a 
Dynamic and Stochastic Inventory Management problem in a Blood Management context. 
It includes a Simulator based on Monte Carlo sampling and a set of different policies available for comparisson.
For more details about the problem details and policies, please refer to
[How to solve Dynamic Resource Allocation problems using RL](https://www.manuelbolivar.com/post/dynamic-resource-allocation/)
blog post.

This project was inspired by Warren Powell's [Approximate Dynamic Programing book](https://www.amazon.com/Approximate-Dynamic-Programmin-Probability-Statistics-dp-047060445X/dp/047060445X/)
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
"scenarios_to_visualize": 0,    # Number of scenarios to be graph and exported to the `figures` folder.
"instance_name": "epoch_15_age_3" # Instance name
```

### Instances

The instances are loaded or generated within the (parameters.py)[parameters.py] file.

In case an instance named as the `instance_name`'s value already exists on the 
(instances/)[instances] folder, it will be loaded. Otherwise, a new instance file 
will be created within the (instances/)[instances] folder. This new instance will use the 
default params defined at the (parameters.py)[parameters.py] file.
Feel free to change the params to generate new instances.

## Resources

+ [How to solve Dynamic Resource Allocation problems using RL](https://www.manuelbolivar.com/post/dynamic-resource-allocation/)
+ [ORF 411 - Sequential Decision Analytics and Modeling.](https://castlelab.princeton.edu/orf-411/) CastleLabs, Princeton.
+ [wbpowell328/stochastic-optimization](https://github.com/wbpowell328/stochastic-optimization) repository.
+ [Approximate Dynamic Programing book](https://www.amazon.com/Approximate-Dynamic-Programmin-Probability-Statistics-dp-047060445X/dp/047060445X/)