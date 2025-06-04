import streamlit as st
from predictor import predict_resources

# Configure the Streamlit app
st.set_page_config(page_title="AI Task Cost Estimator", layout="centered")

# App title and instructions
st.title("🔧 AI Task Cost Prediction & Optimization Engine")
st.markdown("Enter a project description to estimate required resources and cost.")

# Input area for project description
description = st.text_area("📝 Project Description", height=150)

# Role labels with emojis for display
icons = {
    "devs": "👨‍💻 Developers",
    "designers": "🎨 Designers",
    "ai_agents": "🤖 AI Agents",
    "legal_devs": "⚖️ Legal Devs",
    "ai_specialists": "🧠 AI Specialists"
}

# Prediction trigger
if st.button("🚀 Predict Resources"):
    if not description.strip():
        st.warning("Please enter a valid project description.")
    else:
        with st.spinner("Predicting..."):
            resource_counts, intervals, total_cost, r2 = predict_resources(description)

        # Display predicted resources with confidence ranges
        st.subheader("📊 Predicted Resources (with Confidence Ranges)")
        for role, count in resource_counts.items():
            lb, ub = intervals[role]
            role_label = icons.get(role, role.replace('_', ' ').title())
            st.write(f"**{role_label}**: {count} _(range: {lb} - {ub})_")

        # Display total estimated cost
        st.subheader("💰 Estimated Total Cost")
        st.success(f"${total_cost:,.2f}")

        # Display model accuracy
        st.subheader("📈 Model Accuracy")
        st.info(f"Trained model R² Score (pseudo accuracy): {r2:.2f}")
