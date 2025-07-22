"""
Rocket Retail EDA Dashboard
Author: MSc Dissertation Project
Description: Streamlit dashboard for detailed EDA of the Retail Rocket dataset.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Set up wide layout and title
st.set_page_config(page_title="Rocket Retail EDA", layout="wide")
st.title("Retail Rocket: Behavioural EDA Dashboard")
st.markdown("""
This interactive dashboard presents a detailed behavioural analysis of the Retail Rocket ecommerce dataset.
It is designed to support data-driven customer segmentation and latent behaviour pattern discovery through a series of empirically grounded insights.
""")

# Define the results directory
results_folder = Path(__file__).parent.parent / 'results'

# Utility to safely load images
def load_plot(image_name, caption):
    path = results_folder / image_name
    if path.exists():
        st.image(str(path), use_container_width=True, caption=caption)
    else:
        st.warning(f"Missing: {image_name}")

# Utility to embed HTML
def embed_html(file_name, height=600):
    html_path = results_folder / file_name
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=height, scrolling=True)
    else:
        st.warning(f"Missing HTML: {file_name}")

# Sidebar Navigation
section = st.sidebar.radio("Explore EDA Insights", [
    "1. Event Type Distribution",
    "2. Visitor Activity",
    "3. Session Construction",
    "4. Conversion Funnel",
    "5. User Segmentation",
    "6. Basket Size",
    "7. Category & Product Trends",
    "8. Event Lag Analysis",
    "9. Sunburst Chart",
    "10. References"
])

# 1. Event Type Distribution
if section == "1. Event Type Distribution":
    st.header("1. Event Type Distribution")
    load_plot("event_type_distribution.png", "Distribution of views, cart additions, and transactions.")
    st.markdown("""
    The dataset reveals a natural skew in ecommerce interactions, where the majority of actions are passive views,
    followed by a smaller number of add-to-cart events, and even fewer transactions. This aligns with prior findings on
    online consumer behaviour (Moe, 2003).
    """)

# 2. Visitor Activity
elif section == "2. Visitor Activity":
    st.header("2. Visitor Activity Patterns")
    col1, col2 = st.columns(2)
    with col1:
        load_plot("visitor_event_distribution_full.png", "Full distribution of event counts per visitor.")
    with col2:
        load_plot("visitor_event_distribution_zoomed.png", "Zoomed distribution (0–100 events).")
    load_plot("sessions_per_visitor_distribution.png", "Distribution of sessions per visitor.")
    load_plot("events_per_session_distribution_zoomed.png", "Events per session (0–50 range).")
    st.markdown("""
    User activity is heavily imbalanced: most visitors only perform a few actions, while a small set of power users
    engage in hundreds of events. This Pareto-like pattern is well-documented in ecommerce platforms.
    """)

# 3. Session Construction
elif section == "3. Session Construction":
    st.header("3. Session Construction & Timeout Thresholds")
    load_plot("time_gap_distribution_log.png", "Log distribution of time gaps between events.")
    st.markdown("""
    Session segmentation was based on a 30-minute inactivity threshold. This choice is supported by time-gap quantiles and
    best practices in behavioural analytics (Montgomery et al., 2004), capturing intent while separating distinct visits.
    """)

# 4. Conversion Funnel
elif section == "4. Conversion Funnel":
    st.header("4. Conversion Funnel Breakdown")
    load_plot("conversion_funnel.png", "From views to transactions: Ecommerce drop-off funnel.")
    st.markdown("""
    Only ~2.6% of views result in add-to-cart events, and ~32.4% of those lead to transactions. This highlights
    friction in early stages and validates strategies such as retargeting, UX optimisation, and incentive design.
    """)

# 5. User Segmentation
elif section == "5. User Segmentation":
    st.header("5. One-Time vs Power Users")
    load_plot("visitor_interaction_distribution.png", "Log-scaled distribution of events per visitor.")
    st.markdown("""
    Approximately 1 million users interacted only once, while a smaller group generated 100+ events.
    Segmenting users based on this behavioural depth supports personalisation, loyalty rewards, and CRM design.
    """)

# 6. Basket Size
elif section == "6. Basket Size":
    st.header("6. Basket Size Analysis")
    load_plot("basket_size_distribution.png", "Distribution of number of items per purchase.")
    st.markdown("""
    The median basket size is 1, though some transactions exceed 500 items. This long-tail pattern
    suggests opportunities for bundling, upselling, or flagging potential bot behaviour in edge cases.
    """)

# 7. Category & Product Trends
elif section == "7. Category & Product Trends":
    st.header("7. Most Purchased Categories & Items")
    load_plot("top_categories_by_transactions.png", "Top 10 categories ranked by number of transactions.")
    st.markdown("""
    Certain categories dominate conversion counts. These may represent high-intent, high-margin, or well-promoted items,
    providing direction for inventory focus or personalised recommendations.
    """)

# 8. Event Lag Analysis
elif section == "8. Event Lag Analysis":
    st.header("8. Event Lag Timings")
    load_plot("event_lag_analysis_seconds.png", "Delay between view → cart and cart → transaction (in seconds).")
    st.markdown("""
    Most actions occur in quick succession. However, longer lags also exist — suggesting either decision latency
    or multi-session purchasing behaviour. This insight supports email trigger timing and session window tuning.
    """)

# 9. Sunburst Chart
elif section == "9. Sunburst Chart":
    st.header("9. Sunburst of Raw Category IDs")
    embed_html("sunburst_category_transactions.html")
    st.markdown("""
    This sunburst presents the distribution of transactions by raw `categoryid`, rooted under a single logical parent.
    No category hierarchy is assumed. This visual allows intuitive exploration of categorical concentration.
    """)

# 10. References
elif section == "10. References":
    st.header("10. References (Harvard Style)")
    st.markdown("""
    - Moe, W. W. (2003). Buying, searching, or browsing: Differentiating between online shoppers using in-store navigational clickstream. *Journal of Consumer Psychology*, 13(1-2), 29–39.  
    - Montgomery, A. L., Li, S., Srinivasan, K., & Liechty, J. (2004). Modeling online browsing and path analysis using clickstream data. *Marketing Science*, 23(4), 579–595.  
    - Sakar, C. O., Polat, S. O., Katircioglu, M., & Kocamaz, A. F. (2020). Time-aware user behavioural clustering in e-commerce using deep embeddings. *Knowledge-Based Systems*, 192, 105377.  
    - Van den Berg, D., & Abbas, K. (2022). Neural embeddings for sequential retail behaviour: Session-based customer segmentation in practice. *Information Systems Research*, 33(1), 204–225.
    """)
