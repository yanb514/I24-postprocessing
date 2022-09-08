# raw collection
{
"$jsonSchema": {
    "bsonType": "object",
    "properties": {
        "conflicts": {
            "bsonType": "array",
            "items": {
                "bsonType": "array"
                },
            "description": "[other_id_it_conflicts_with, time_for_conflicts]"
            },
        "feasibility": {
            "bsonType": "object",
            "properties": {
                "acceleration": {
                    "bsonType": "float",
                    "description": "float 0-1, percentage of time acceleration is out of bound (+/- 10 ft/s2)"
                    }
                },
                "backward": {
                    "bsonType": "float",
                    "description": "float 0-1, percentage of time a car travels backwards"
                },
                "conflict": {
                    "bsonType": "float",
                    "description": "float 0-1, percentage of time a car conflicts with any other"
                },
                "distance": {
                    "bsonType": "float",
                    "description": "float 0-1, distance traveled over x-range"
                },
                "rotation": {
                    "bsonType": "float",
                    "description": "float 0-1, percentage of time rotation is out of bound (30 degree)"
                },
            }
        }
    }
}

# postprocessed collection

{
"$jsonSchema": {
    "bsonType": "object",
    "properties": {
        "merge": {
            "bsonType": "object",
            "properties": {
                "merged_ids":{
                    "bsonType": "array",
                    },
                "conflicts": {
                    "bsonType": "array",
                    "items": {
                        "bsonType": "array"
                        },
                    "description": "[other_id_it_conflicts_with, time_for_conflicts]"
                    },
                "feasibility": {
                    "bsonType": "object",
                    "properties": {
                        "acceleration": {
                            "bsonType": "float",
                            "description": "float 0-1, percentage of time acceleration is out of bound (+/- 10 ft/s2)"
                            }
                        },
                        "backward": {
                            "bsonType": "float",
                            "description": "float 0-1, percentage of time a car travels backwards"
                        },
                        "conflict": {
                            "bsonType": "float",
                            "description": "float 0-1, percentage of time a car conflicts with any other"
                        },
                        "distance": {
                            "bsonType": "float",
                            "description": "float 0-1, distance traveled over x-range"
                        },
                        "rotation": {
                            "bsonType": "float",
                            "description": "float 0-1, percentage of time rotation is out of bound (30 degree)"
                        },
                    }
                }
            },
        
        "stitch": {
            "bsonType": "object",
            "properties": {
                "fragment_ids":{
                    "bsonType": "array"
                    },
                "conflicts": {
                    "bsonType": "array",
                    "items": {
                        "bsonType": "array"
                        },
                    "description": "[other_id_it_conflicts_with, time_for_conflicts]"
                    },
                "feasibility": {
                    "bsonType": "object",
                    "properties": {
                        "acceleration": {
                            "bsonType": "float",
                            "description": "float 0-1, percentage of time acceleration is out of bound (+/- 10 ft/s2)"
                            }
                        },
                        "backward": {
                            "bsonType": "float",
                            "description": "float 0-1, percentage of time a car travels backwards"
                        },
                        "conflict": {
                            "bsonType": "float",
                            "description": "float 0-1, percentage of time a car conflicts with any other"
                        },
                        "distance": {
                            "bsonType": "float",
                            "description": "float 0-1, distance traveled over x-range"
                        },
                        "rotation": {
                            "bsonType": "float",
                            "description": "float 0-1, percentage of time rotation is out of bound (30 degree)"
                        },
                    }
                }
            },
        
        "reconcile": {
            "bsonType": "object",
            "properties": {
                "conflicts": {
                    "bsonType": "array",
                    "items": {
                        "bsonType": "array"
                        },
                    "description": "[other_id_it_conflicts_with, time_for_conflicts]"
                    },
                "feasibility": {
                    "bsonType": "object",
                    "properties": {
                        "acceleration": {
                            "bsonType": "float",
                            "description": "float 0-1, percentage of time acceleration is out of bound (+/- 10 ft/s2)"
                            }
                        },
                        "backward": {
                            "bsonType": "float",
                            "description": "float 0-1, percentage of time a car travels backwards"
                        },
                        "conflict": {
                            "bsonType": "float",
                            "description": "float 0-1, percentage of time a car conflicts with any other"
                        },
                        "distance": {
                            "bsonType": "float",
                            "description": "float 0-1, distance traveled over x-range"
                        },
                        "rotation": {
                            "bsonType": "float",
                            "description": "float 0-1, percentage of time rotation is out of bound (30 degree)"
                        }
                    }
                }
            }
        }
    }
}





