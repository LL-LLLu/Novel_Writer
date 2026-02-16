"""Standard prompts for novel generation benchmarking."""

BENCHMARK_PROMPTS = [
    {
        "id": "continuation",
        "name": "Story Continuation",
        "prompt": "Continue the following story:\n\nThe old library held secrets that no one dared to speak of. When Sarah found the hidden room behind the shelves, she",
        "category": "continuation",
    },
    {
        "id": "dialogue",
        "name": "Dialogue Generation",
        "prompt": "Write a tense dialogue between two characters who have just discovered a betrayal:\n\n",
        "category": "dialogue",
    },
    {
        "id": "description",
        "name": "Scene Description",
        "prompt": "Describe a mysterious forest at twilight, focusing on sensory details:\n\n",
        "category": "description",
    },
    {
        "id": "character",
        "name": "Character Introduction",
        "prompt": "Introduce a complex antagonist who believes they are doing the right thing:\n\n",
        "category": "character",
    },
    {
        "id": "action",
        "name": "Action Sequence",
        "prompt": "Write an intense chase scene through a crowded marketplace:\n\n",
        "category": "action",
    },
]
