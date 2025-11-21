hyperparameters ={
    "antmaze-large-navigate-singletask-v0": {
        "alpha_actor": 10.0,
        "alpha" : 0.1,
        "num_samples":4,

        "mfql":{
            "num_samples":4,
        },
        "dmfql":{
            "alpha" : 0.03,
            "num_samples":4,
        },
        "dmfrebrac":{
            "alpha": 0.03,
            "alpha_critic":0.01,
            "num_samples":4,
        },
    },
    "antmaze-giant-navigate-singletask-v0": {
        "alpha_actor": 10.0,
        "alpha" : 0.1,
        "num_samples": 4,

        "mfql":{
            "num_samples":4,
        },
        "dmfql":{
            "alpha" : 0.1,
            "num_samples":4,
        },
        "dmfrebrac":{
            "alpha": 0.1,
            "alpha_critic":0.01,
            "num_samples":4,
        },
    },
    "humanoidmaze-medium-navigate-singletask-v0": {
        "alpha_actor": 30.0,
        "alpha" : 0.1,
        "num_samples":32,

        "diql": {
            "alpha" : 0.3,
            "alpha_actor": 10.0,
        },        
        "mfql":{
            "num_samples":32,
        },
        "dmfql":{
            "alpha" : 0.1,
            "num_samples":32,
        },
        "dmfrebrac":{
            "alpha": 0.1,
            "alpha_critic":0.01,
            "num_samples": 32,
        },
    },
    "humanoidmaze-large-navigate-singletask-v0": {
        "alpha_actor": 30.0,
        "alpha":0.03,
        "num_samples":8,

        "diql": {
            "alpha_actor": 10.0,
            "alpha": 0.1,
        },
        "mfql":{
            "num_samples":16,
        },
        "dmfql":{
            "num_samples":16,
            "alpha" : 0.03,
        },
        "dmfrebrac":{
            "alpha": 0.03,
            "alpha_critic":0.01,
            "num_samples": 16,
        },
    },
    "antsoccer-arena-navigate-singletask-v0": {
        "alpha_actor": 10.0,
        "alpha":0.1,
        "num_samples":16,

        "diql": {
            "alpha_actor": 1.0,
            "alpha" : 0.1
        },
        "mfql":{
            "num_samples":16,
        },
        "dmfql":{
            "num_samples":16,
            "alpha" : 0.1,
        },
        "dmfrebrac":{
            "alpha": 0.1,
            "alpha_critic":0.01,
            "num_samples": 16,
        },
    },
    "cube-single-play-singletask-v0": {
        "alpha_actor": 300.0,
        "alpha":0.03,
        "num_samples":32,


        "diql": {
            "alpha_actor": 1.0,
            "alpha":0.03,
        },
        "mfql":{
            "num_samples":2,
        },
        "dmfql":{
            "num_samples":2,
            "alpha" : 0.03,
        },
        "dmfrebrac":{
            "alpha": 0.03,
            "alpha_critic":0.01,
            "num_samples": 2,
        },
    },
    "cube-double-play-singletask-v0": {
        "alpha_actor": 300.0,
        "alpha":0.1,
        "num_samples":16,

        "diql": {
            "alpha_actor": 0.3,
            "alpha" : 0.1,
        },
        "mfql":{
            "num_samples":4,
        },
        "dmfql":{
            "num_samples":4,
            "alpha" : 0.03,
        },
        "dmfrebrac":{
            "alpha": 0.03,
            "alpha_critic":0.01,
            "num_samples": 4,
        },
    },
    "scene-play-singletask-v0": {
        "alpha_actor": 300.0,
        "alpha":0.1,
        "num_samples":32,

        "diql": {
            "alpha_actor": 10.0,
            "alpha": 0.1,
        },
        "mfql":{
            "num_samples":4,
        },
        "dmfql":{
            "num_samples":4,
            "alpha" : 0.1,
        },
        "dmfrebrac":{
            "alpha": 0.1,
            "alpha_critic":0.01,
            "num_samples": 4,
        },
    },
    "puzzle-3x3-play-singletask-v0": {
        "alpha_actor": 1000.0,
        "alpha":0.03,
        "num_samples":4,

        "diql": {
            "alpha_actor": 10.0,
            "alpha": 0.1,
        },
        "mfql":{
            "num_samples":2,
        },
        "dmfql":{
            "num_samples":2,
            "alpha" : 0.03,
        },
        "dmfrebrac":{
            "alpha": 0.03,
            "alpha_critic":0.01,
            "num_samples": 2,
        },
    },
    "puzzle-4x4-play-singletask-v0": {
        "alpha_actor": 1000.0,
        "alpha":0.03,
        "num_samples":64,

        "diql": {
            "alpha_actor": 3.0,
            "alpha":0.1,
        },
        "mfql":{
            "num_samples":4,
        },
        "dmfql":{
            "num_samples":4,
            "alpha" : 0.03,
        },
        "dmfrebrac":{
            "alpha": 0.03,
            "alpha_critic":0.01,
            "num_samples": 4,
        },
    },
    "pen-expert-v1": {
        "alpha_actor": 3000.0,
        "alpha":0.3,
        "num_samples":64,
        "diql": {
            "alpha": 0.3,
        },
        "mfql":{
            "num_samples":32,
        },
        "dmfql":{
            "num_samples":32,
            "alpha":0.03,
        },
        "dmfrebrac":{
            "num_samples":32,
            "alpha_critic":0.01,
        },
    },
    "pen-human-v1": {
        "alpha_actor": 10000.0,
        "alpha":0.001,
        "num_samples":8,
        "diql": {
            "alpha": 0.1,
        },
        "mfql":{
            "num_samples":4,
        },
        "dmfql":{
            "num_samples":4,
            "alpha": 0.00003,
        },
        "dmfrebrac":{
            "num_samples":4,
            "alpha_critic":0.01,
        },
    },
    "pen-cloned-v1": {
        "alpha":0.0001,
        "num_samples":16,
        "alpha_actor": 10000.0,
        "diql": {
            "alpha": 0.0001,
        },
        "mfql":{
            "num_samples":32,
        },
        "dmfql":{
            "num_samples":32,
            "alpha": 0.000003,
        },
        "dmfrebrac":{
            "num_samples":32,
            "alpha_critic":0.01,
        },
        
    },
    "door-expert-v1": {
        "alpha_actor": 30000.0,
        "alpha":0.003,
        "num_samples":1,
        "diql": {
            "alpha": 0.00003,
        },
        "mfql":{
            "num_samples":16,
        },
        "dmfql":{
            "num_samples":16,
            "alpha": 0.0001,
        },
        "dmfrebrac":{
            "num_samples":16,
            "alpha_critic":0.01,
        },

    },
    "hammer-expert-v1": {
        "alpha_actor": 30000.0,
        "alpha":0.000003,
        "num_samples":1,
        "diql": {
            "alpha": 0.03,
        },
        "mfql":{
            "num_samples":4,
        },
        "dmfql":{
            "num_samples":4,
            "alpha": 0.03,
        },
        "dmfrebrac":{
            "num_samples":4,
            "alpha_critic":0.01,
        },
    },
    "relocate-expert-v1": {
        "alpha_actor": 30000.0,
        "alpha":0.0001,
        "num_samples":2,

        "diql": {
            "alpha": 0.01,
        }, 
        "mfql":{
            "num_samples":4,
        },
        "dmfql":{
            "num_samples":4,
            "alpha": 0.0001,
        },
        "dmfrebrac":{
            "num_samples":4,
            "alpha_critic":0.01,
        },
    },

}