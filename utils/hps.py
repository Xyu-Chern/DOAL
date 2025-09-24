hyperparameters ={
    "antmaze-large-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0,
        },
        "diql": {
            "alpha_actor": 10.0,
        },
        "alpha_actor": 10.0,
        "alpha":0.1,
        "num_samples":4,
    },
    "antmaze-giant-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0,
        },
        "diql": {
            "alpha_actor": 10.0,
        },
        "alpha_actor": 10.0,
        "alpha":0.1,
        "num_samples":4,
    },
    "humanoidmaze-medium-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0,
        },
        "diql": {
            "alpha_actor": 10.0,
        },
        "alpha_actor": 30.0,
        "alpha":0.3,
        "num_samples":32,
    },
    "humanoidmaze-large-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0,
        },
        "diql": {
            "alpha_actor": 10.0,
        },
        "alpha_actor": 30.0,
        "alpha":0.1,
        "num_samples":8,
    },
    "antsoccer-arena-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 1.0
        },
        "diql": {
            "alpha_actor": 1.0,
        },
         "alpha_actor": 10.0,
        "alpha":0.03,
        "num_samples":4,
    },
    "cube-single-play-singletask-v0": {
        "iql": {
            "alpha_actor": 1.0
        },
        "diql": {
            "alpha_actor": 1.0,
        },
        "alpha_actor": 300.0,
        "alpha":0.03,
        "num_samples":32,
    },
    "cube-double-play-singletask-v0": {
        "iql": {
            "alpha_actor": 0.3
        },
        "diql": {
            "alpha_actor": 0.3,
        },
        "alpha_actor": 300.0,
        "alpha":0.1,
        "num_samples":16,
    },
    "scene-play-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0
        },
        "diql": {
            "alpha_actor": 10.0,
        },
        "alpha_actor": 300.0,
        "alpha":0.03,
        "num_samples":32,
        
    },
    "puzzle-3x3-play-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0
        },
        "diql": {
            "alpha_actor": 10.0,
        },
        "alpha_actor": 1000.0,
        "alpha":0.03,
        "num_samples":8,

    },
    "puzzle-4x4-play-singletask-v0": {
        "iql": {
            "alpha_actor": 3.0
        },
        "diql": {
            "alpha_actor": 3.0,
        },
        "alpha_actor": 1000.0,
        "alpha":0.1,
        "num_samples":32,
    },
    "antmaze-umaze-v2": {
        "fql": {
            "alpha_actor": 10.0
        },
        "trigflow": {
            "alpha_actor": 10.0
        },
        "rebrac": {
            "alpha_actor": 0.003
        },
        "alpha_critic": 0.002,
        
        "drebrac": {
            "alpha_actor": 0.003,
            "alpha_critic": 0.002
        }
    },
    "antmaze-umaze-diverse-v2": {
        "fql": {
            "alpha_actor": 10.0
        },
        "trigflow": {
            "alpha_actor": 10.0
        },
        "rebrac": {
            "alpha_actor": 0.003
        },
        "alpha_critic": 0.001,
        
        "drebrac": {
            "alpha_actor": 0.003,
            "alpha_critic": 0.001
        }
    },
    "antmaze-medium-play-v2": {
        "fql": {
            "alpha_actor": 10.0
        },
        "trigflow": {
            "alpha_actor": 10.0
        },
        "rebrac": {
            "alpha_actor": 0.001
        },
        "alpha_critic": 0.0005,
        
        "drebrac": {
            "alpha_actor": 0.001,
            "alpha_critic": 0.0005
        }
    },
    "antmaze-medium-diverse-v2": {
        "fql": {
            "alpha_actor": 10.0
        },
        "trigflow": {
            "alpha_actor": 10.0
        },
        "rebrac": {
            "alpha_actor": 0.001
        },
        "alpha_critic": 0.0,
        "drebrac": {
            "alpha_actor": 0.001,
            "alpha_critic": 0.0
        }
    },
    "pen-expert-v1": {
        "alpha_actor": 3000.0,
        "iql": {
            "alpha_actor": 0.5
        },
        "alpha":0.3,
        "num_samples":128,
    },
    "pen-human-v1": {
        "alpha_actor": 10000.0,
        "iql": {
            "alpha_actor": 0.5
        },
        "alpha":0.03,
        "num_samples":8,
        
    },
    "pen-cloned-v1": {
        "alpha_actor": 10000.0,
        "iql": {
            "alpha_actor": 0.5
        },
        "alpha":0.03,
        "num_samples":32,
        
    },
    "door-expert-v1": {
            "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },
        "alpha":0.03,
        "num_samples":2,
    },
    "door-cloned-v1": {
        "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },  
        "alpha":0.03,
        "num_samples":2,
    },
    "door-human-v1": {
        "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },  
        "alpha":0.03,
        "num_samples":8,
    },
    "hammer-expert-v1": {
        "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },
        "alpha":0.1,
        "num_samples":2,
    },
    "hammer-cloned-v1": {
        "alpha_actor": 10000.0,
        "iql": {
            "alpha_actor": 0.5
        },  
        "alpha":0.3,
        "num_samples":4,
    },
    "hammer-human-v1": {
        "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },      
        "alpha":0.03,
        "num_samples":8,
    },
    "relocate-expert-v1": {
        "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },    
        "alpha":0.03,
        "num_samples":2,
    },
    "relocate-human-v1": {
        "alpha_actor": 10000.0,
        "iql": {
            "alpha_actor": 0.5
        },
        "alpha":0.1,
        "num_samples":64,
    },
    "relocate-cloned-v1": {    
        "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },
        "alpha":0.1,
        "num_samples":128,
    },

}