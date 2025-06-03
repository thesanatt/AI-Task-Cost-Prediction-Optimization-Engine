# Estimated cost per resource per week (in USD)
RESOURCE_COSTS = {
    "devs": 2000,
    "designers": 1500,
    "ai_agents": 2500,
    "legal_devs": 2200,
    "ai_specialists": 3000
}

# Duration assumption (in weeks)
PROJECT_DURATION_WEEKS = 4

def estimate_cost(resource_counts: dict) -> float:
    total = 0
    for role, count in resource_counts.items():
        rate = RESOURCE_COSTS.get(role, 0)
        total += count * rate * PROJECT_DURATION_WEEKS
    return total