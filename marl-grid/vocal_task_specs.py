TASK_VOCABS = {
    "find-goal": {
        "slot_names": ["intent_entity", "direction", "distance"],
        "slot_vocabs": [
            [
                "I found the goal",
                "I reached the goal and stopped",
                "I hit a wall / obstacle",
                "I am exploring blindly",
            ],
            [
                "to my North",
                "to my South",
                "to my East",
                "to my West",
                "to my North-East",
                "to my North-West",
                "to my South-East",
                "to my South-West",
                "right here",
            ],
            [
                "exactly 1 step away",
                "exactly 2 steps away",
                "exactly 3 steps away",
                "out of my sight",
            ],
        ],
    },
    "red-blue-doors": {
        "slot_names": ["actor", "operation", "target"],
        "slot_vocabs": [
            ["I", "You"],
            ["wait at", "approach", "opened"],
            ["Red Door", "Blue Door", "Nowhere"],
        ],
    },
    "overcooked": {
        "slot_names": ["action_intent", "entity", "location_status"],
        "slot_vocabs": [
            [
                "I am picking up",
                "I am putting down",
                "I am waiting for",
                "Please get",
            ],
            [
                "an Onion",
                "a Plate",
                "the Soup",
                "Nothing / Empty hands",
            ],
            [
                "from the Dispenser",
                "to the Pot",
                "to the Serving Station",
                "at the Counter",
            ],
        ],
    },
}


TASK_OUTPUTS = {
    "find-goal": "text_embeddings_find_goal.pt",
    "red-blue-doors": "text_embeddings_red_blue_doors.pt",
    "overcooked": "text_embeddings_overcooked.pt",
}


def get_supported_tasks():
    return tuple(TASK_VOCABS.keys())


def get_task_vocab(task):
    if task not in TASK_VOCABS:
        raise KeyError("Unknown task: {}".format(task))
    return TASK_VOCABS[task]
