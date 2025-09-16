hyperparameters ={
    "antmaze-large-navigate-singletask-v0": {
        "alpha_actor": 10.0,
        "iql": {
            "alpha_actor": 10.0,
        },
        "diql": {
            "alpha_actor": 10.0,
        },
        "alpha_critic": 0.01,
        "delta":1.0,
        "alpha":3.0,
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
        "delta":1.0,
        "alpha": 3.0,
    },
    "humanoidmaze-medium-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0,
        },
        "fql": {
            "alpha_actor": 30.0,
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
        "delta":0.3,
        "alpha": 100.0,
    },
    "humanoidmaze-large-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0,
        },
        "fql": {
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
        "delta":1.0,
        "alpha": 30.0,
    },
    "antsoccer-arena-navigate-singletask-v0": {
        "iql": {
            "alpha_actor": 1.0
        },
        "fql": {
            "alpha_actor": 10.0,
        },
        "diql": {
            "alpha_actor": 1.0,
        },
        "rebrac": {
            "alpha_actor": 0.01
        },
        "alpha_critic": 0.01,
        "delta":0.1,
        "alpha": 100.0,
    },
    "cube-single-play-singletask-v0": {
        "iql": {
            "alpha_actor": 1.0
        },
        "fql": {
            "alpha_actor": 300.0,
        },
        "diql": {
            "alpha_actor": 1.0,
        },
        "rebrac": {
            "alpha_actor": 1.0
        },
        "alpha_critic": 1.0,
        "delta":1.0,
        "alpha": 300.0,
    },
    "cube-double-play-singletask-v0": {
        "iql": {
            "alpha_actor": 0.3
        },
        "fql": {
            "alpha_actor": 300.0,
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
        "delta":0.1,
        "alpha": 300.0,
    },
    "scene-play-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0
        },
        "fql": {
            "alpha_actor": 300.0,
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
        "delta":1.0,
        "alpha": 300.0,
    },
    "puzzle-3x3-play-singletask-v0": {
        "iql": {
            "alpha_actor": 10.0
        },
        "fql": {
            "alpha_actor": 1000.0,
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
        "delta": 1.0,
        "alpha": 1000.0,

    },
    "puzzle-4x4-play-singletask-v0": {
        "iql": {
            "alpha_actor": 3.0
        },
        "diql": {
            "alpha_actor": 3.0,
        },
        "fql": {
            "alpha_actor": 1000.0,
        },
        "rebrac": {
            "alpha_actor": 0.3
        },
        "alpha_critic": 0.01,
        "drebrac": {
            "alpha_actor": 0.3,
            "alpha_critic": 0.01
        },
        "delta": 3.0,
        "alpha": 1000.0,
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
        "alpha": 0.003,
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
        "alpha": 0.003,
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
        "alpha": 0.001,
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
        "alpha": 0.001,
        "drebrac": {
            "alpha_actor": 0.001,
            "alpha_critic": 0.0
        }
    },
    "pen-expert-v1": {
        "alpha": 10000.0,
    },
    "door-expert-v1": {
        "alpha": 10000.0,
    },
    "hammer-expert-v1": {
        "alpha": 1000.0,
    },
    "relocate-expert-v1": {
        "alpha": 1000.0,
    },
    # "visual-cube-single-play-singletask-v0": {
    #     "iql": {
    #         "alpha_actor": 1.0
    #     },
    #     "diql": {
    #         "alpha_actor": 1.0
    #     },
    #     "fql": {
    #         "alpha_actor": 300.0
    #     },
    #     "trigflow": {
    #         "alpha_actor": 300.0
    #     }
    # },
    # "visual-cube-double-play-singletask-v0": {
    #     "iql": {
    #         "alpha_actor": 0.3
    #     },
    #     "diql": {
    #         "alpha_actor": 0.3
    #     },
    #     "fql": {
    #         "alpha_actor": 100.0
    #     },
    #     "trigflow": {
    #         "alpha_actor": 100.0
    #     }
    # },
    # "visual-scene-play-singletask-v0": {
    #     "iql": {
    #         "alpha_actor": 10.0
    #     },
    #     "diql": {
    #         "alpha_actor": 10.0
    #     },
    #     "fql": {
    #         "alpha_actor": 100.0
    #     },
    #     "trigflow": {
    #         "alpha_actor": 100.0
    #     }
    # },
    # "visual-puzzle-3x3-play-singletask-v0": {
    #     "iql": {
    #         "alpha_actor": 10.0
    #     },
    #     "diql": {
    #         "alpha_actor": 10.0
    #     },
    #     "fql": {
    #         "alpha_actor": 300.0
    #     },
    #     "trigflow": {
    #         "alpha_actor": 300.0
    #     }
    # },
    # "visual-puzzle-4x4-play-singletask-v0": {
    #     "iql": {
    #         "alpha_actor": 3.0
    #     },
    #     "diql": {
    #         "alpha_actor": 3.0
    #     },
    #     "fql": {
    #         "alpha_actor": 300.0
    #     },
    #     "trigflow": {
    #         "alpha_actor": 300.0
    #     }
    # }
}