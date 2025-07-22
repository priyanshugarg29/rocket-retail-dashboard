# Rocket Retail EDA Dashboard

This project presents an interactive Streamlit dashboard developed for advanced exploratory data analysis (EDA) of the Retail Rocket ecommerce dataset. It is part of an MSc Business Analytics dissertation that investigates latent customer behaviour segmentation using neural embeddings and unsupervised clustering techniques.

## Project Aim

The primary objective is to extract interpretable and actionable insights from real-world online retail interaction data. The dashboard is designed to support both exploratory understanding and downstream behavioural modelling, laying the groundwork for data-driven customer segmentation and personalised marketing strategies.

## Dataset Description

The analysis is based on the publicly available [Retail Rocket dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset), which includes:
- `events.csv`: logs of user interactions (view, add-to-cart, transaction)
- `item_properties_part1.csv` and `part2.csv`: item-level metadata (e.g. categoryid, availability)
- `category_tree.csv`: sparse hierarchical structure for categories

The dataset spans millions of user sessions and tens of thousands of items, reflecting natural customer journeys in an online retail setting.

## Dashboard Features

The Streamlit dashboard integrates key exploratory insights, including:

- **Event Type Distribution**: Breakdown of views, cart additions, and transactions
- **Visitor Engagement**: Histograms of events per visitor and sessions per user
- **Session Construction**: Gap-based session ID generation using 30-minute timeout
- **Funnel Analysis**: View ➝ Add-to-Cart ➝ Transaction flow with drop-off interpretation
- **User Segmentation**: One-timers vs power users distribution
- **Basket Size Analysis**: Quantity of items per purchase
- **Product & Category Trends**: Most purchased items and categories
- **Event Lag Analysis**: Time taken from view to cart and purchase
- **Sunburst Chart**: Interactive breakdown of transactions across raw `categoryid`s

All visual artefacts are derived directly from the EDA notebook and rendered using `matplotlib` and `plotly`.

## Methodology

Key stages include:

1. **Data Preprocessing**: Timestamp conversion, merging item properties
2. **Sessionization**: Time-gap-based segmentation of raw events into sessions
3. **Feature Engineering**: Construction of session-level behavioural sequences
4. **Statistical EDA**: Distributional analysis, funnel metrics, lag patterns
5. **Export and Deployment**: Generation of all outputs in `results/` for reproducibility

## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rocket-retail-dashboard.git
   cd rocket-retail-dashboard
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Launch the dashboard:
   ```
   streamlit run dashboard/app.py
   ```

Ensure all plots and summary outputs are available in the `results/` directory.

## Academic References (Harvard Style)

- Moe, W. W. (2003). Buying, searching, or browsing: Differentiating between online shoppers using in-store navigational clickstream. *Journal of Consumer Psychology*, 13(1-2), 29–39.
- Montgomery, A. L., Li, S., Srinivasan, K., & Liechty, J. (2004). Modeling online browsing and path analysis using clickstream data. *Marketing Science*, 23(4), 579–595.
- Sakar, C. O., Polat, S. O., Katircioglu, M., & Kocamaz, A. F. (2020). Time-aware user behavioural clustering in e-commerce using deep embeddings. *Knowledge-Based Systems*, 192, 105377.
- Van den Berg, D., & Abbas, K. (2022). Neural embeddings for sequential retail behaviour: Session-based customer segmentation in practice. *Information Systems Research*, 33(1), 204–225.

## License

This repository is for academic and research purposes only.