def extract_features(description: str) -> dict:
    desc = description.lower()
    roles = ["devs", "designers", "ai_agents", "legal_devs", "ai_specialists"]

    keyword_scores = {
        "mobile": {"devs": 2, "designers": 1},
        "bank": {"devs": 2, "legal_devs": 1, "ai_specialists": 1},
        "ai": {"ai_agents": 2, "ai_specialists": 2},
        "chatbot": {"ai_agents": 2},
        "login": {"devs": 1},
        "ecommerce": {"devs": 2, "designers": 1},
        "landing page": {"designers": 1},
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
        "secure": {"devs": 1},
        "ml": {"ai_specialists": 2, "ai_agents": 1},
        "nlp": {"ai_specialists": 1, "ai_agents": 2}
    }

    score = {role: 0 for role in roles}

    for keyword, weights in keyword_scores.items():
        if keyword in desc:
            for role, w in weights.items():
                score[role] += w

    # Set a max cap per role (avoid over-inflation)
    for role in score:
        score[role] = min(score[role], 2)

    # Ensure at least 1 dev if any other resource is present
    if sum(score.values()) > 0 and score["devs"] == 0:
        score["devs"] = 1

    return score