hyperparameters ={
    "antmaze-large-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0,
        },
        "diql": {
            "alpha_actor": 10.0,
        },
        "alpha_critic": 0.01,
        "alpha_actor": 10.0,
        "alpha":0.1,
    },
    "antmaze-giant-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0,
        },
        "diql": {
            "alpha_actor": 10.0,
        },
        "rebrac": {
            "alpha_actor": 0.003
        },
        "alpha_critic": 0.01,
        "drebrac": {
            "alpha_actor": 0.003,
            "alpha_critic": 0.01
        },
        "alpha_actor": 10.0,
        "alpha":0.1,
        
        
    },
    "humanoidmaze-medium-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0,
        },
        "diql": {
            "alpha_actor": 10.0,
        },
        "rebrac": {
            "alpha_actor": 0.01
        },
        "alpha_critic": 0.01,
        "drebrac": {
            "alpha_actor": 0.01,
            "alpha_critic": 0.01
        },
        "alpha_actor": 30.0,
        "alpha":0.2,
    },
    "humanoidmaze-large-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0,
        },
        "diql": {
            "alpha_actor": 10.0,
        },
        "rebrac": {
            "alpha_actor": 0.01
        },
        "alpha_critic": 0.01,
        "drebrac": {
            "alpha_actor": 0.01,
            "alpha_critic": 0.01
        },
        "alpha_actor": 30.0,
        "alpha":0.2,
    },
    "antsoccer-arena-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 1.0
        },
        "diql": {
            "alpha_actor": 1.0,
        },
        "rebrac": {
            "alpha_actor": 0.01
        },
        "alpha_critic": 0.01,
         "alpha_actor": 10.0,
        "alpha":0.05,
    },
    "cube-single-play-singletask-v0": {
        "iql": {
            "alpha_actor": 1.0
        },
        "diql": {
            "alpha_actor": 1.0,
        },
        "rebrac": {
            "alpha_actor": 1.0
        },
        "alpha_critic": 1.0,
        "alpha_actor": 300.0,
        "alpha":0.05,
    },
    "cube-double-play-singletask-v0": {
        "iql": {
            "alpha_actor": 0.3
        },
        "diql": {
            "alpha_actor": 0.3,
        },
        "rebrac": {
            "alpha_actor": 1.0
        },
        "alpha_critic": 1.0,
        "drebrac": {
            "alpha_actor": 1.0,
            "alpha_critic": 1.0
        },
        "alpha_actor": 300.0,
        "alpha":0.1,
    },
    "scene-play-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0
        },
        "fql": {
            "alpha_actor": 300.0
        },
        "diql": {
            "alpha_actor": 10.0,
        },
        "rebrac": {
            "alpha_actor": 0.1
        },
        "alpha_critic": 0.01,
        "drebrac": {
            "alpha_actor": 0.1,
            "alpha_critic": 0.01
        },
        "alpha_actor": 100.0,
        "alpha":0.1,
        
    },
    "puzzle-3x3-play-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0
        },
        "diql": {
            "alpha_actor": 10.0,
        },
        "rebrac": {
            "alpha_actor": 0.3
        },
        "alpha_critic": 0.01,
        "drebrac": {
            "alpha_actor": 0.3,
            "alpha_critic": 0.01
        },
            "alpha_actor": 1000.0,
        "alpha":0.1,

    },
    "puzzle-4x4-play-singletask-v0": {
        "iql": {
            "alpha_actor": 3.0
        },
        "diql": {
            "alpha_actor": 3.0,
        },
        "rebrac": {
            "alpha_actor": 0.3
        },
        "alpha_critic": 0.01,
        "drebrac": {
            "alpha_actor": 0.3,
            "alpha_critic": 0.01
        },
        "alpha_actor": 1000.0,
        "alpha":0.1,
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
    },
    "pen-human-v1": {
        "alpha_actor": 10000.0,
        "iql": {
            "alpha_actor": 0.5
        },
        
    },
    "pen-cloned-v1": {
            "alpha_actor": 10000.0,
        "iql": {
            "alpha_actor": 0.5
        },
        
    },
    "door-expert-v1": {
            "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },
    },
    "door-cloned-v1": {
            "alpha_actor": 30000.0,
            "alpha_actor": 0.5
        },  
    },
    "door-human-v1": {
            "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },  
    },
    "hammer-expert-v1": {
            "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },
    },
    "hammer-cloned-v1": {
            "alpha_actor": 10000.0,
        "iql": {
            "alpha_actor": 0.5
        },  
    },
    "hammer-human-v1": {
            "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },      
    },
    "relocate-expert-v1": {
            "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },    
    },
    "relocate-human-v1": {
            "alpha_actor": 10000.0,
        "iql": {
            "alpha_actor": 0.5
        },
    },
    "relocate-cloned-v1": {    
        "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },
    },

}


