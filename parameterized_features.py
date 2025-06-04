def extract_features(description: str) -> dict:
    '''
    Converts a project description into a dictionary of resource role scores
    based on keyword matching (non-strict). Uses a custom override for "landing page".
    '''
    desc = description.lower()
    roles = ["devs", "designers", "ai_agents", "legal_devs", "ai_specialists"]

    keyword_scores = {
        "landing page": {"designers": 1, "devs": 1},
        "mobile": {"devs": 2, "designers": 1},
        "bank": {"devs": 2, "legal_devs": 1, "ai_specialists": 1},
        "marketplace": {"devs": 2, "designers": 1},
        "ai": {"ai_agents": 2, "ai_specialists": 2},
        "chatbot": {"ai_agents": 2},
        "ml": {"ai_specialists": 2, "ai_agents": 1},
        "nlp": {"ai_specialists": 1, "ai_agents": 2},
        "login": {"devs": 1},
        "ecommerce": {"devs": 2, "designers": 1},
        "api": {"devs": 2},
        "dashboard": {"designers": 1},
        "real-time": {"ai_agents": 1},
        "compliance": {"legal_devs": 1},
        "insurance": {"legal_devs": 1},
        "hospital": {"legal_devs": 1, "ai_specialists": 1},
        "data": {"ai_agents": 1, "ai_specialists": 1},
        "analytics": {"ai_agents": 1},
        "review": {"legal_devs": 1},
        "model": {"ai_specialists": 2},
        "fraud": {"ai_specialists": 1, "legal_devs": 1},
        "finance": {"legal_devs": 1},
        "documentation": {"designers": 1},
        "reporting": {"designers": 1},
        "cloud": {"devs": 2, "ai_specialists": 1},
        "prediction": {"ai_specialists": 2},
        "secure": {"devs": 1}
    }

    score = {role: 0 for role in roles}
    matched = False

    for keyword, weights in keyword_scores.items():
        if keyword in desc:
            matched = True
            for role, w in weights.items():
                score[role] += w

    # Special override for any appearance of "landing page"
    if "landing page" in desc:
        return {"devs": 1, "designers": 1, "ai_agents": 0, "legal_devs": 0, "ai_specialists": 0}

    # Fallback: if keywords matched and no devs were assigned
    if matched and sum(score.values()) > 0 and score["devs"] == 0:
        score["devs"] = 1

    return score
