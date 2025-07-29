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
    "10. Executive Summary",
    "11. References"
])

# 1. Event Type Distribution
if section == "1. Event Type Distribution":
    st.header("1. Event Type Distribution")
    load_plot("event_type_distribution.png", "Distribution of views, cart additions, and transactions.")
    st.markdown("""
   Out of approximately 2.75 million total events:

- View events account for ~96.7% of all interactions (2.66M), indicating high browsing activity.

- Add-to-cart events make up only ~2.5%, reflecting selective product engagement.

- Transaction events represent just ~0.8%, consistent with typical ecommerce funnel conversion rates.

This sharp drop-off from view to purchase aligns with known online consumer behaviour patterns, where most sessions are exploratory in nature and only a small fraction culminates in a purchase (Moe, 2003; Sakar et al., 2020).

Such a distribution underscores the need to model non-purchasing behaviour with equal rigour—embedding sequences based solely on conversions would neglect the rich intent signals present in cart additions and repeated views. The segmentation framework must therefore incorporate complete session sequences to reveal behavioural diversity beyond monetary action.
""")

# 2. Visitor Activity
elif section == "2. Visitor Activity":
    st.header("2. Visitor Activity Patterns")
    col1, col2 = st.columns(2)
    with col1:
        load_plot("visitor_event_distribution_full.png", "Full distribution of event counts per visitor.")
    with col2:
        load_plot("visitor_event_distribution_zoomed.png", "Zoomed distribution (0–100 events).")

    st.markdown("""For understanding interaction variability, we visualised the distribution of event counts per visitor—both across the full range and restricted to a 0–100 event window for clearer inspection.

**Key Insights:**
- The first plot (full range) reveals a highly right-skewed distribution, with the vast majority of users interacting only a handful of times. However, a small minority (long tail) exhibit extremely high engagement, with the most active visitor recording 7,757 interactions. This reflects Pareto-like behaviour, consistent with ecommerce platforms where a minority of users contribute disproportionately to interaction volume (Montgomery et al., 2004).

- The second plot (0–100 events) highlights that over 95% of visitors engage in fewer than 100 events, with a sharp drop-off visible beyond 10–20 interactions. This suggests that while the dataset includes power users, it is dominated by low-engagement or one-time visitors, which is a common trait in publicly available ecommerce datasets (Van den Berg & Abbas, 2022; Moe, 2003).

**Relevance:**
- This behaviour supports our methodological choice to model at the session level rather than user level. Aggregating across users would conflate diverse behaviours and dilute signals from brief but meaningful interactions (Sakar et al., 2020).

- It also validates the need for unsupervised segmentation, as traditional cohorting based on fixed thresholds (e.g., top 10% by engagement) would miss the behavioural heterogeneity evident in the long tail.

In line with behavioural economics, this skew reflects diverse browsing and buying patterns—ranging from impulse drop-ins to persistent comparers—and forms the basis for downstream segmentation via neural embeddings and clustering.""")
  
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
    The lag histograms reveal an extremely tight clustering near zero minutes for both transitions:

- Over 90% of add-to-cart and transaction events occur immediately after the preceding event. This reflects intent continuity—users who add items to their cart or purchase tend to act in a single behavioural session without prolonged gaps.

- A very small number of user-item paths demonstrate delayed decision-making, which may indicate deliberation, price sensitivity, or return visits—these are outliers but still relevant for nuanced cluster formation.

These patterns confirm the short-attention span nature of ecommerce behaviour and align with findings by Moe (2003) and Sakar et al. (2020), who argue that most conversions follow a fast funnel collapse once purchase intent crystallises.

**Relevance to Segmentation:**

- This motivates our choice of session-level modelling and justifies the use of 30-minute session gaps (Montgomery et al., 2004).

- Lag durations are stored for use in cluster profiling—e.g., clusters with longer cart-to-transaction times may reflect more hesitant or price-sensitive users.
    """)

    st.markdown(""" | Metric    | Value (Seconds) | Interpretation                                                                     |
| --------- | --------------- | ---------------------------------------------------------------------------------- |
| 25th %ile | 38              | 25% of interactions happen within 38 seconds of each other — high browsing density |
| Median    | 136             | Half of all event pairs occur within just 2.3 minutes                              |
| 75th %ile | 2,449           | 75% of users return within 40 minutes — within browsing intent window              |
| 90th %ile | 263,524         | A long tail begins—10% of transitions span over 73 hours                           |
| 95th %ile | 1,190,249       | 5% of transitions exceed 13.8 days                                                 |
| 99th %ile | 5,160,078       | Extreme lags observed—often due to multiple visits across sessions                 |
| Maximum   | 11,787,451      | Indicates multi-week session gaps (approx. 136 days)                               |

These statistics illustrate a heavy-tailed distribution of time gaps. While the majority of user actions are clustered within a short time span, a small fraction of interactions are spaced days or even weeks apart. This supports the rationale behind adopting a 30-minute inactivity threshold for defining session boundaries, as proposed in prior literature (Montgomery et al., 2004; Moe, 2003; Google Analytics, 2023). The 75th percentile lies well within this range, ensuring that true behavioural sessions are captured without splitting meaningful flows or merging unrelated ones.

Such quantile-based diagnostics are especially critical in ecommerce datasets where repeated visits, multi-device access, and asynchronous behaviour can distort naive time assumptions. Understanding the actual time gap distribution strengthens the reliability of downstream sessionisation and embedding steps.
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

# 10. Executive Summary
elif section == "10. Executive Summary":
    st.header("10. Executive Summary")
    st.markdown("""
    This dashboard provides a comprehensive behavioural breakdown of the Retail Rocket ecommerce dataset, structured around the customer's interaction journey.

    Beginning with an analysis of event types, it becomes evident that online shopping behaviour is skewed towards browsing, with significantly fewer add-to-cart and purchase actions. This foundational imbalance is further reflected in user engagement, where most visitors exhibit minimal interaction while a small subset engages deeply and repeatedly. Session segmentation, based on data-driven time gap analysis, enables meaningful delineation of these behaviours into discrete user journeys.

    Conversion funnel insights show pronounced drop-offs at early stages, underscoring the importance of improving mid-funnel engagement. Basket size distributions further reveal typical one-product purchases with rare but significant outliers, suggesting differing intent or purchase contexts.

    User segmentation patterns follow a Pareto-like shape, validating the use of clustering techniques for high-impact targeting. Category and product trends identify key conversion-driving items, while lag analysis uncovers both impulsive and deliberative customer behaviours. Finally, the sunburst chart visualises the distribution of transactions across raw category IDs without inferring any hierarchy, ensuring analytical integrity.

    Together, these findings build a cohesive behavioural profile of Rocket Retail users, guiding both strategic decisions and the development of downstream machine learning models.
    """)

    # Offer PDF download
    pdf_path = results_folder / "executive_summary.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            st.download_button("Download Executive Summary (PDF)", f, file_name="executive_summary.pdf")
    else:
        st.info("PDF version of the executive summary is not available.")

elif section == "11. References":
    st.header("11. References")
    st.markdown("""
    - Moe, W. W. (2003). Buying, searching, or browsing: Differentiating between online shoppers using in-store navigational clickstream. *Journal of Consumer Psychology*, 13(1-2), 29–39.  
    - Montgomery, A. L., Li, S., Srinivasan, K., & Liechty, J. (2004). Modeling online browsing and path analysis using clickstream data. *Marketing Science*, 23(4), 579–595.  
    - Sakar, C. O., Polat, S. O., Katircioglu, M., & Kocamaz, A. F. (2020). Time-aware user behavioural clustering in e-commerce using deep embeddings. *Knowledge-Based Systems*, 192, 105377.  
    - Van den Berg, D., & Abbas, K. (2022). Neural embeddings for sequential retail behaviour: Session-based customer segmentation in practice. *Information Systems Research*, 33(1), 204–225.
    """)
