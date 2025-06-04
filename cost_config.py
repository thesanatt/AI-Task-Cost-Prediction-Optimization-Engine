# Weekly cost (in USD) for each type of resource
ROLE_COSTS = {
    "devs": 2000,
    "designers": 1500,
    "ai_agents": 2500,
    "legal_devs": 2200,
    "ai_specialists": 3000
}

# Default project duration assumption (in weeks)
PROJECT_DURATION_WEEKS = 4

def estimate_cost(resource_counts: dict) -> float:
    """
    Calculate the total project cost based on resource allocation and duration.

    Parameters:
        resource_counts (dict): Mapping of resource roles to predicted counts

    Returns:
        float: Total estimated project cost in USD
    """
    total = 0
    for role, count in resource_counts.items():
        rate = ROLE_COSTS.get(role, 0)
        total += count * rate * PROJECT_DURATION_WEEKS
    return total
