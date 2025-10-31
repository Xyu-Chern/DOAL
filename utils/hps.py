hyperparameters ={
    "antmaze-large-navigate-singletask-v0": {
        "alpha_actor": 10.0,
        "alpha":0.03,
        "num_samples":4,

        "iql": {
            "alpha_actor": 10.0,
        },
        "diql": {
            "alpha_actor": 10.0,
            "alpha" : 0.1,
        },
        "dtrigflow": {
            "alpha" : 0.1,
        },
        "mfql":{
            "num_samples":4,
        },
        "dmfql":{
            "num_samples":4,
            "alpha" : 0.03,
        },
        "drebrac":{
            "alpha_actor":0.003,
            "alpha_critic":0.01,
            "alpha":0.1,
        },
    },
    "antmaze-giant-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0,
        },
        "diql": {
            "alpha_actor": 10.0,
            "alpha" : 0.1,
        },
        "alpha_actor": 10.0,
        "alpha":0.03,
        "num_samples":4,
        "mfql":{
            "num_samples":4,
        },
        "dmfql":{
            "num_samples":4,
            "alpha" : 0.1,
        },
        "drebrac":{
            "alpha_actor":0.003,
            "alpha_critic":0.01,
            "alpha":0.1,
        },
    },
    "humanoidmaze-medium-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0,
        },
        "diql": {
            "alpha_actor": 10.0,
            "alpha" : 0.3,
        },
        "alpha_actor": 30.0,
        "alpha":0.1,
        "num_samples":32,

        "mfql":{
            "num_samples":32,
        },
        "dmfql":{
            "num_samples":32,
            "alpha" : 0.1,
        },
        "drebrac":{
            "alpha_actor":0.01,
            "alpha_critic":0.01,
            "alpha":0.3,
        },
    },
    "humanoidmaze-large-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0,
        },
        "diql": {
            "alpha_actor": 10.0,
            "alpha": 0.1,
        },
        "alpha_actor": 30.0,
        "alpha":0.03,
        "num_samples":8,
        "mfql":{
            "num_samples":16,
        },
        "dmfql":{
            "num_samples":16,
            "alpha" : 0.03,
        },
        "drebrac":{
            "alpha_actor":0.01,
            "alpha_critic":0.01,
            "alpha":0.1,
        },
    },
    "antsoccer-arena-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 1.0
        },
        "diql": {
            "alpha_actor": 1.0,
            "alpha" : 0.1
        },
         "alpha_actor": 10.0,
        "alpha":0.1,
        "num_samples":16,
        "mfql":{
            "num_samples":16,
        },
        "dmfql":{
            "num_samples":16,
            "alpha" : 0.1,
        },
        "drebrac":{
            "alpha_actor":0.01,
            "alpha_critic":0.01,
            "alpha":0.1,
        },
    },
    "cube-single-play-singletask-v0": {
        "iql": {
            "alpha_actor": 1.0
        },
        "diql": {
            "alpha_actor": 1.0,
            "alpha":0.03,
        },
        "alpha_actor": 300.0,
        "alpha":0.03,
        "num_samples":32,
        "mfql":{
            "num_samples":2,
        },
        "dmfql":{
            "num_samples":2,
            "alpha" : 0.03,
        },
        "drebrac":{
            "alpha_actor":1.0,
            "alpha_critic":0.0,
            "alpha":0.03,
        },
    },
    "cube-double-play-singletask-v0": {
        "iql": {
            "alpha_actor": 0.3
        },
        "diql": {
            "alpha_actor": 0.3,
            "alpha" : 0.1,
        },
        "alpha_actor": 300.0,
        "alpha":0.1,
        "num_samples":16,
        "mfql":{
            "num_samples":4,
        },
        "dmfql":{
            "num_samples":4,
            "alpha" : 0.03,
        },
        "drebrac":{
            "alpha_actor":0.1,
            "alpha_critic":0.0,
            "alpha":0.1,
        },
    },
    "scene-play-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0
        },
        "diql": {
            "alpha_actor": 10.0,
            "alpha": 0.1,
        },
        "alpha_actor": 300.0,
        "alpha":0.1,
        "num_samples":32,
        "mfql":{
        "num_samples":4,
        },
        "dmfql":{
        "num_samples":4,
        "alpha" : 0.1,
        },
        "drebrac":{
            "alpha_actor":0.1,
            "alpha_critic":0.01,
            "alpha":0.1,
        },
        
    },
    "puzzle-3x3-play-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0
        },
        "diql": {
            "alpha_actor": 10.0,
            "alpha": 0.1,
        },
        "alpha_actor": 1000.0,
        "alpha":0.03,
        "num_samples":4,
        "mfql":{
        "num_samples":2,
        },
        "dmfql":{
        "num_samples":2,
        "alpha" : 0.03,
        },
        "drebrac":{
            "alpha_actor":0.3,
            "alpha_critic":0.01,
            "alpha":0.1,
        },

    },
    "puzzle-4x4-play-singletask-v0": {
        "iql": {
            "alpha_actor": 3.0
        },
        "diql": {
            "alpha_actor": 3.0,
            "alpha":0.1,
        },
        "alpha_actor": 1000.0,
        "alpha":0.03,
        "num_samples":64,
        "mfql":{
        "num_samples":4,
        },
        "dmfql":{
        "num_samples":4,
        "alpha" : 0.03,
        },
        "drebrac":{
            "alpha_actor":0.3,
            "alpha_critic":0.01,
            "alpha":0.1,
        },
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
        "num_samples":64,
        "mfql":{
            "num_samples":32,
        },
        "dmfql":{
            "num_samples":32,
        },
    },
    "pen-human-v1": {
        "alpha_actor": 10000.0,
        "iql": {
            "alpha_actor": 0.5
        },
        "alpha":0.001,
        "num_samples":8,
        "mfql":{
            "num_samples":4,
        },
        "dmfql":{
            "num_samples":4,
        },
        
    },
    "pen-cloned-v1": {
        "alpha_actor": 10000.0,
        "iql": {
            "alpha_actor": 0.5
        },
        "alpha":0.0001,
        "num_samples":16,
        "mfql":{
            "num_samples":32,
        },
        "dmfql":{
            "num_samples":32,
        },
        
    },
    "door-expert-v1": {
            "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },
        "alpha":0.003,
        "num_samples":1,
        "mfql":{
            "num_samples":16,
        },
        "dmfql":{
            "num_samples":16,
        },

    },
    "door-cloned-v1": {
        "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },  
        "alpha":0.003,
        "num_samples":64,
    },
    "door-human-v1": {
        "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },  
        "alpha":0.03,
        "num_samples":2,
    },
    "hammer-expert-v1": {
        "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },
        "alpha":0.000003,
        "num_samples":1,
        "mfql":{
            "num_samples":4,
        },
        "dmfql":{
            "num_samples":4,
        },
    },
    "hammer-cloned-v1": {
        "alpha_actor": 10000.0,
        "iql": {
            "alpha_actor": 0.5
        },  
        "alpha":0.3,
        "num_samples":8,
    },
    "hammer-human-v1": {
        "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },      
        "alpha":0.03,
        "num_samples":4,
    },
    "relocate-expert-v1": {
        "alpha_actor": 30000.0,
        "iql": {
            "alpha_actor": 0.5
        },    
        "alpha":0.0001,
        "num_samples":2,
        "mfql":{
            "num_samples":4,
        },
        "dmfql":{
            "num_samples":4,
        },

    },
    "relocate-human-v1": {
        "alpha_actor": 10000.0,
        "iql": {
            "alpha_actor": 0.5
        },
        "alpha":0.1,
        "num_samples":32,
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