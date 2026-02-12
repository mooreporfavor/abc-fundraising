import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Theme Colors ---
CUSTOM_COLORWAY = [
    "#2D9B8F", "#D4A574", "#C85A54", "#1A5F57", 
    "#E8B4A0", "#4DBFB3", "#8B6F47", "#A0D5D0", "#D9A89F"
]

# --- Page Configuration ---
st.set_page_config(
    page_title="Donor Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Theme & CSS ---
# Inject custom CSS based on the requested design
st.markdown("""
<style>
    /* Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    /* General Defaults */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        color: #3D3D3D;
        background-color: #FFFFFF;
    }

    /* Titles & Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1A5F57; /* Dark Teal */
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
    }
    
    /* Hero/Top Section Override */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Metric Cards (KPIs) */
    div[data-testid="stMetric"], div[data-testid="metric-container"] {
        background-color: #F8F6F3; /* Off-white/Beige */
        border: 1px solid #E0D5CC;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s ease-in-out;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 12px rgba(0,0,0,0.1);
    }
    
    div[data-testid="stMetricLabel"] > label {
        color: #1A5F57 !important;
        font-size: 14px !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stMetricValue"] {
        color: #2D9B8F !important; /* Primary Teal */
        font-size: 28px !important;
        font-weight: 700;
    }
    
    .st-emotion-cache-1r6slb0 { /* Helper text in metrics if present */
        color: #8B6F47 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F3F1ED; /* Slightly darker beige */
        border-right: 1px solid #E0D5CC;
    }
    
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #2D9B8F; 
    }

    /* Buttons */
    .stButton > button {
        background-color: #2D9B8F;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #1A5F57;
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 4px 4px 0px 0px;
        color: #3D3D3D;
        font-weight: 500;
        border-bottom: 2px solid #E0D5CC;
    }

    .stTabs [aria-selected="true"] {
        background-color: #F8F6F3;
        border-bottom: 3px solid #2D9B8F;
        color: #1A5F57;
        font-weight: 700;
    }

    /* Info Alerts */
    .stAlert {
        background-color: #E8F4F2;
        border: 1px solid #2D9B8F;
        color: #1A5F57;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #F8F6F3;
        border-radius: 4px;
        color: #3D3D3D;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E0D5CC;
        border-radius: 4px;
    }

</style>
""", unsafe_allow_html=True)


# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("task_1_processed_v2.csv", dtype={'Donor_ID': str})
        
        # Date Conversions
        date_cols = ['First_Gift_Date', 'Last_Contact_Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Numeric Conversions
        numeric_cols = [
            'Lifetime_Giving', 'Giving_Last_24_Months', 'Annualized_Lifetime_Value', 
            'Recent_Annualized_Giving', 'Touchpoints_Last_12_Months', 'Engagement_Velocity',
            'Churn_Risk_Score', 'Contact_Recency_Score'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Ensure string columns are strings
        str_cols = ['Geography', 'Industry', 'Relationship_Stage', 'Churn_Risk_Category', 'Drift_Status', 'Assigned_RM', 'Notes']
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '')

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No data loaded. Please check if 'task_1_processed_v2.csv' exists.")
    st.stop()


# --- Sidebar Filters ---
with st.sidebar:
    st.header("Dashboard Controls")
    
    # "Show Ghosts Only" Toggle
    show_ghosts = st.toggle("Show Ghosts Only", value=False, help="Filter for High Value (> $500k Lifetime) but Zero Recent Giving")
    
    # Global Filters
    st.subheader("Global Filters")
    
    # Assigned RM
    all_rms = sorted(df['Assigned_RM'].unique().tolist())
    selected_rms = st.multiselect("Assigned RM", options=all_rms, default=all_rms)
    
    # Industry
    all_industries = sorted(df['Industry'].unique().tolist())
    selected_industries = st.multiselect("Industry", options=all_industries, default=all_industries)
    
    # Geography
    all_geos = sorted(df['Geography'].unique().tolist())
    selected_geos = st.multiselect("Geography", options=all_geos, default=all_geos)

    # Relationship Stage
    all_stages = sorted(df['Relationship_Stage'].unique().tolist())
    selected_stages = st.multiselect("Relationship Stage", options=all_stages, default=all_stages)

    # Churn Risk Category
    all_risks = sorted(df['Churn_Risk_Category'].unique().tolist())
    selected_risks = st.multiselect("Churn Risk Category", options=all_risks, default=all_risks)


# --- Apply Filters ---
filtered_df = df.copy()

if show_ghosts:
    # "Ghosts": Lifetime > 500k AND Giving Last 24 Months == 0
    filtered_df = filtered_df[
        (filtered_df['Lifetime_Giving'] > 500000) & 
        (filtered_df['Giving_Last_24_Months'] == 0)
    ]

if selected_rms:
    filtered_df = filtered_df[filtered_df['Assigned_RM'].isin(selected_rms)]

if selected_industries:
    filtered_df = filtered_df[filtered_df['Industry'].isin(selected_industries)]

if selected_geos:
    filtered_df = filtered_df[filtered_df['Geography'].isin(selected_geos)]
    
if selected_stages:
    filtered_df = filtered_df[filtered_df['Relationship_Stage'].isin(selected_stages)]

if selected_risks:
    filtered_df = filtered_df[filtered_df['Churn_Risk_Category'].isin(selected_risks)]


# --- Top-Level Layout ---

# Hero Section
with st.container():
    st.markdown("""
    <div style="background-color: #1A5F57; padding: 20px; border-radius: 8px; margin-bottom: 20px; color: white;">
        <h1 style="color: white; margin-bottom: 0px;">Donor Insights Dashboard</h1>
        <p style="font-size: 16px; opacity: 0.9; margin-top: 5px;">
            Comprehensive view of donor engagement, lifetime value, and portfolio health.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Context Line
st.markdown(f"**Data loaded:** {len(filtered_df):,} / {len(df):,} rows | **Last Refreshed:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

st.divider()

# --- KPI Topline Metrics (The "Pulse") ---
# Pipeline Velocity: Sum of Giving_Last_24_Months
# At-Risk Capital: Sum of Lifetime_Giving for donors flagged as "High Risk / Dormant"
# Dormancy Rate: Percentage of portfolio with Last_Contact_Date > 6 months ago

kpi1, kpi2, kpi3 = st.columns(3)

# 1. Pipeline Velocity
pipeline_velocity = filtered_df['Giving_Last_24_Months'].sum()
kpi1.metric("Pipeline Velocity (Last 24M)", f"${pipeline_velocity:,.0f}")

# 2. At-Risk Capital
# Filter logic: Drift_Status == "High Risk / Dormant" OR Churn_Risk_Category == "High Risk" (User spec says Drift_Status)
at_risk_df = filtered_df[filtered_df['Drift_Status'] == "High Risk / Dormant"]
at_risk_capital = at_risk_df['Lifetime_Giving'].sum()
kpi2.metric("At-Risk Capital", f"${at_risk_capital:,.0f}", help="Lifetime giving of donors marked 'High Risk / Dormant'")

# 3. Dormancy Rate
# Last Contact > 6 months ago
six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
dormant_count = filtered_df[filtered_df['Last_Contact_Date'] < six_months_ago].shape[0]
total_count = len(filtered_df)
dormancy_rate = (dormant_count / total_count * 100) if total_count > 0 else 0
kpi3.metric("Dormancy Rate (>6mo No Contact)", f"{dormancy_rate:.1f}%")

st.divider()

# --- Data Cards Component ---
# Card 1: "Total Donors"
# Card 2: "Total Lifetime Giving"
# Card 3: "Avg Lifetime Value"
# Card 4: "High Risk Donors"
# Card 5: "Avg Engagement Velocity"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Donors", f"{len(filtered_df):,}")
c2.metric("Total Lifetime Giving", f"${filtered_df['Lifetime_Giving'].sum():,.0f}")
c3.metric("Avg Lifetime Value", f"${filtered_df['Annualized_Lifetime_Value'].mean():,.0f}")
c4.metric("High Risk Donors", f"{filtered_df[filtered_df['Churn_Risk_Category'] == 'High Risk'].shape[0]:,}")
c5.metric("Avg Engagement Velocity", f"{filtered_df['Engagement_Velocity'].mean():.2f}")


st.divider()

# --- Main Visuals Tabs ---
tab1, tab2, tab3 = st.tabs(["Efficiency Matrix", "Ghost Table", "Deep Dives"])

with tab1:
    st.subheader("Efficiency Matrix: Effort vs. Revenue")
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        color_by = st.selectbox("Color By:", ["Drift_Status", "Relationship_Stage", "Assigned_RM"], index=0)
    with col_ctrl2:
        size_by = st.selectbox("Bubble Size:", ["Lifetime_Giving", "Annualized_Lifetime_Value", "Recent_Annualized_Giving"], index=0)
        
    s_chart_df = filtered_df.copy()
    # Filter out NaNs for plot
    s_chart_df = s_chart_df.dropna(subset=['Touchpoints_Last_12_Months', 'Giving_Last_24_Months', size_by])
    
    if not s_chart_df.empty:
        fig_matrix = px.scatter(
            s_chart_df,
            x="Touchpoints_Last_12_Months",
            y="Giving_Last_24_Months",
            size=size_by,
            color=color_by,
            hover_data=["Donor_ID", "Notes"],
            title="Efficiency Matrix",
            color_discrete_sequence=CUSTOM_COLORWAY,
            template="plotly_white"
        )
        
        # Custom Colors override if needed based on "Theme Details"
        # Since Plotly templates are distinct, let's try to stick to the requested palette in the config or passed manually
        
        # Update layout
        fig_matrix.update_layout(
            height=600,
            xaxis_title="Touchpoints Last 12 Months (Effort)",
            yaxis_title="Giving Last 24 Months (Revenue)",
            yaxis_tickprefix="$",
        )
        st.plotly_chart(fig_matrix, use_container_width=True)
    else:
        st.info("No data available for Efficiency Matrix with current filters.")

with tab2:
    st.subheader("Ghost Table: High Risk / Dormant Donors")
    st.markdown("Monday morning reactivation hit list. Filters logic: `Drift_Status` == 'High Risk / Dormant', sorted by Lifetime Giving.")
    
    ghost_df = filtered_df[filtered_df['Drift_Status'] == "High Risk / Dormant"].copy()
    ghost_df = ghost_df.sort_values(by="Lifetime_Giving", ascending=False)
    
    # Columns to display
    display_cols = ["Donor_ID", "Assigned_RM", "Lifetime_Giving", "Last_Contact_Date", "Notes"]
    
    # Format Date
    ghost_df['Last_Contact_Date'] = ghost_df['Last_Contact_Date'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(
        ghost_df[display_cols],
        column_config={
            "Lifetime_Giving": st.column_config.NumberColumn("Lifetime Giving", format="$%d"),
            "Notes": st.column_config.TextColumn("Notes", width="large")
        },
        use_container_width=True,
        height=600
    )

with tab3:
    st.subheader("Deep Dive Analysis")
    
    # 1. Lifetime Giving by First Gift Year
    st.markdown("#### Lifetime Giving by First Gift Year")
    
    dd_col1, dd_col2 = st.columns(2)
    with dd_col1:
        granularity = st.selectbox("Granularity", ["Year", "Quarter"])
    with dd_col2:
        aggregation = st.selectbox("Aggregation", ["Sum", "Mean", "Median", "Count"], index=1) # Default Mean/Average
    
    # Process
    if 'First_Gift_Date' in filtered_df.columns:
        cohort_df = filtered_df.copy().dropna(subset=['First_Gift_Date'])
        if granularity == "Year":
            cohort_df['Period'] = cohort_df['First_Gift_Date'].dt.year
        else:
            cohort_df['Period'] = cohort_df['First_Gift_Date'].dt.to_period('Q').astype(str)
            
        agg_map = {"Sum": "sum", "Mean": "mean", "Median": "median", "Count": "count"}
        cohort_agg = cohort_df.groupby('Period')['Lifetime_Giving'].agg(agg_map[aggregation]).reset_index()
        
        fig_cohort = px.line(
            cohort_agg, 
            x='Period', 
            y='Lifetime_Giving',
            markers=True,
            title=f"Lifetime Giving by First Gift {granularity}",
            color_discrete_sequence=["#2D9B8F"], # Teal
            template="plotly_white"
        )
        fig_cohort.update_layout(hovermode="x unified")
        st.plotly_chart(fig_cohort, use_container_width=True)
        
    st.divider()
    
    # 2. Industry Comparison
    st.markdown("#### Industry Lifetime Giving Comparison")
    
    ind_col1, ind_col2, ind_col3 = st.columns(3)
    with ind_col1:
        ind1 = st.selectbox("Chart 1 Industry", options=all_industries, index=0 if len(all_industries) > 0 else 0)
    with ind_col2:
         # Try to pick a different default if possible
        default_idx = 1 if len(all_industries) > 1 else 0
        ind2 = st.selectbox("Chart 2 Industry", options=all_industries, index=default_idx)
    with ind_col3:
        metric_ind = st.selectbox("Metric", ["Sum", "Mean", "Count"])
        
    # Helper to plot industry
    def plot_industry(ind_name, metric):
        ind_data = filtered_df[filtered_df['Industry'] == ind_name]
        if ind_data.empty:
            st.write(f"No data for {ind_name}")
            return
            
        agg_func = {"Sum": "sum", "Mean": "mean", "Count": "count"}
        # Group by Donor ID? The request says "X: Donor_ID".
        # If we aggregate by Donor ID, "Sum of Lifetime Giving" implies sum per donor? 
        # But Lifetime Giving IS per donor. So it's just plotting the values.
        # "Sum" probably means we just plot the value of Lifetime_Giving for each donor.
        # IF there were multiple rows per donor, we'd aggregate. Assuming one row per donor here.
        
        # Sort desc
        ind_data_sorted = ind_data.sort_values(by="Lifetime_Giving", ascending=False)
        
        # Limit to top 50 for readability if there are many? Request doesn't specify limit but sidebar "Top N" exists elsewhere.
        # Let's just plot all or top 20
        ind_data_sorted = ind_data_sorted.head(50) 
        
        y_col = "Lifetime_Giving"
        title_metric = "Lifetime Giving"
        if metric == "Count":
             # Count of donors? No, X is Donor_ID. 
             # If metric is "Count", maybe user means something else?
             # "Count of Donors" in a bar chart where X is Donor_ID doesn't make sense (always 1).
             # Maybe X should be Industry? But the chart title is "Industry Lifetime Giving Comparison". 
             # Request: "X: Donor_ID, Y: Depends on metric". 
             # If Metric is "Count of Donors", this implies we are aggregating NOT by donor ID but just showing a single bar?
             # Or maybe "Count of gifts"? We don't have that detail.
             # I will skip "Count" for per-donor plot or treating it as "1". 
             pass
        
        fig = px.bar(
            ind_data_sorted,
            x="Donor_ID",
            y=y_col,
            title=f"{ind_name}: {metric} {title_metric}",
            color_discrete_sequence=["#D4A574"], # Tan
            template="plotly_white"
        )
        return fig

    c_ind1, c_ind2 = st.columns(2)
    with c_ind1:
        st.plotly_chart(plot_industry(ind1, metric_ind), use_container_width=True)
    with c_ind2:
        st.plotly_chart(plot_industry(ind2, metric_ind), use_container_width=True)

    st.divider()

    # 3. Donor LTV Projection Model
    st.markdown("#### Donor LTV Projection Model")
    st.info("Input global filters first, then refine with chart-specific controls below.")
    
    ltv_col1, ltv_col2, ltv_col3, ltv_col4 = st.columns(4)
    with ltv_col1:
        proj_years = st.selectbox("Projection Period (Years)", [3, 5, 10], index=1)
        discount_rate_options = [0.0, 0.03, 0.05, 0.07]
        discount_rate = st.selectbox("Discount Rate", discount_rate_options, index=1, format_func=lambda x: f"{x:.0%}")
    with ltv_col2:
        scenario = st.selectbox("Scenario", ["Conservative", "Realistic", "Optimistic"], index=1)
        thresh_options = [0, 50000, 100000, 250000, 500000]
        ltv_threshold = st.selectbox("LTV Threshold", thresh_options, index=2, format_func=lambda x: f"${x:,}")
    with ltv_col3:
        # Defaults to Active/Mid/Late if they exist
        default_stages = [s for s in ["Active", "Mid", "Late"] if s in all_stages]
        ltv_stages = st.multiselect("Relationship Stage (LTV Model)", options=all_stages, default=default_stages)
    with ltv_col4:
        color_ltv = st.selectbox("Color By", ["None", "Donor_ID"], index=0)

    # Filter Data for LTV
    # Use global filtered_df as base
    ltv_df = filtered_df.copy()
    
    # Apply specific LTV filters
    if ltv_stages:
        ltv_df = ltv_df[ltv_df['Relationship_Stage'].isin(ltv_stages)]
    
    if ltv_threshold > 0:
        ltv_df = ltv_df[ltv_df['Lifetime_Giving'] >= ltv_threshold]
        
    if not ltv_df.empty:
        # Computations
        # historical_growth_rate
        ltv_df['historical_growth_rate'] = np.where(
            ltv_df['Annualized_Lifetime_Value'] > 0,
            (ltv_df['Recent_Annualized_Giving'] - ltv_df['Annualized_Lifetime_Value']) / ltv_df['Annualized_Lifetime_Value'],
            0
        )
        
        # adjusted_growth_rate
        ltv_df['adjusted_growth_rate'] = (
            ltv_df['historical_growth_rate'] * 
            (1 - ltv_df['Churn_Risk_Score'] / 100) * 
            (ltv_df['Engagement_Velocity'] / 10)
        )
        
        # Scenario Logic
        if scenario == "Conservative":
            growth_rate = ltv_df['adjusted_growth_rate'].quantile(0.25)
        elif scenario == "Optimistic":
            growth_rate = ltv_df['adjusted_growth_rate'].quantile(0.75)
        else: # Realistic
            growth_rate = ltv_df['adjusted_growth_rate'].mean()
            
        # Projection
        rate_modifier = (1 + growth_rate) / (1 + discount_rate)
        
        if rate_modifier == 1:
            term_multiplier = proj_years
        else:
            term_multiplier = rate_modifier * (1 - rate_modifier**proj_years) / (1 - rate_modifier)
            
        ltv_df['Projected_Value'] = ltv_df['Recent_Annualized_Giving'] * term_multiplier
        ltv_df['Total_Projected_LTV'] = ltv_df['Lifetime_Giving'] + ltv_df['Projected_Value']
        
        # Sorting & limit
        ltv_plot_df = ltv_df.sort_values(by="Total_Projected_LTV", ascending=False).head(50) 
        
        # Color
        c_arg = "Donor_ID" if color_ltv == "Donor_ID" else None
        
        fig_ltv = px.bar(
            ltv_plot_df,
            x="Donor_ID",
            y="Total_Projected_LTV",
            color=c_arg,
            title=f"Projected Donor LTV ({scenario} Scenario, {proj_years}-Year Horizon)",
            color_discrete_sequence=["#2D9B8F"],
            template="plotly_white"
        )
        st.plotly_chart(fig_ltv, use_container_width=True)
        
        # Summary Table
        st.dataframe(
            ltv_df[["Donor_ID", "Lifetime_Giving", "Total_Projected_LTV", "adjusted_growth_rate"]],
            column_config={
                "Lifetime_Giving": st.column_config.NumberColumn("Current LTV", format="$%d"),
                "Total_Projected_LTV": st.column_config.NumberColumn(f"Projected {proj_years}-Year Value", format="$%d"),
                "adjusted_growth_rate": st.column_config.NumberColumn("Adj Growth Rate", format="%.2f%%")
            },
            use_container_width=True
        )
        
    else:
        st.info("No donors match the selected filters for LTV Model.")

    st.divider()

    # 4. Top Donors by Lifetime Giving
    st.markdown("#### Top Donors by Lifetime Giving")
    
    td_col1, td_col2 = st.columns(2)
    with td_col1:
        top_n = st.selectbox("Top N Donors", [5, 10, 20, 50, 100], index=1)
    with td_col2:
        top_agg = st.selectbox("Aggregation Method (Top Donors)", ["Sum", "Average", "Median"], index=0)
        
    # Data
    top_df = filtered_df.copy()
    if not top_df.empty:
        agg_map_top = {"Sum": "sum", "Average": "mean", "Median": "median"}
        # For 'Top Donors', typically we want individual donors. Aggregation only matters if duplicates exist.
        top_grouped = top_df.groupby("Donor_ID")["Lifetime_Giving"].agg(agg_map_top[top_agg]).reset_index()
        top_grouped = top_grouped.sort_values(by="Lifetime_Giving", ascending=False).head(top_n)
        
        fig_top = px.bar(
            top_grouped,
            y="Donor_ID",
            x="Lifetime_Giving",
            orientation='h',
            title=f"Top {top_n} Donors by Lifetime Giving",
            color_discrete_sequence=["#2D9B8F"],
            template="plotly_white"
        )
        fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.write("No data available.")

    st.divider()

    # 5. Lifetime Giving by Industry
    st.markdown("#### Lifetime Giving by Industry")
    
    li_row1_col1, li_row1_col2 = st.columns(2)
    with li_row1_col1:
        li_agg = st.selectbox("Aggregation Method (Industry)", ["Sum", "Average", "Count", "Max"], index=0)
    with li_row1_col2:
        # Note: 'Other' logic in pandas is handled below
        li_geo_options = ["All Geographies", "US – CA", "US – NY", "UK", "Canada", "Other"] 
        # Ensure these match actual data or use generic
        li_geo = st.selectbox("Geography Filter", li_geo_options, index=0)
        
    # Filtering
    li_df = filtered_df.copy()
    if li_geo != "All Geographies":
        if li_geo == "Other":
            # Exclude specific major ones
            exclude_geos = ["US – CA", "US – NY", "UK", "Canada"]
            # Normalize dashes just in case
            li_df = li_df[~li_df['Geography'].isin(exclude_geos)]
        else:
            li_df = li_df[li_df['Geography'] == li_geo]
    
    if not li_df.empty:
        agg_funcs_li = {"Sum": "sum", "Average": "mean", "Count": "count", "Max": "max"}
        li_grouped = li_df.groupby("Industry")["Lifetime_Giving"].agg(agg_funcs_li[li_agg]).reset_index()
        li_grouped = li_grouped.sort_values(by="Lifetime_Giving", ascending=False)
        
        fig_li = px.bar(
            li_grouped,
            x="Industry",
            y="Lifetime_Giving",
            text="Lifetime_Giving",
            title=f"Lifetime Giving by Industry ({li_agg})",
            color_discrete_sequence=["#D4A574"],
            template="plotly_white"
        )
        # Format text
        fig_li.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        st.plotly_chart(fig_li, use_container_width=True)
    else:
        st.write(f"No data for Geography: {li_geo}")
    
    st.divider()
    
    # 6. Recent Giving Trend by Last Contact
    st.markdown("#### Recent Giving Trend by Last Contact")
    
    rg_col1, rg_col2 = st.columns(2)
    with rg_col1:
        rg_gran = st.selectbox("Time Granularity", ["Month", "Quarter"], index=0)
    with rg_col2:
        rg_agg = st.selectbox("Aggregation Method (Trend)", ["Average", "Sum", "Median"], index=0)
        
    rg_df = filtered_df.copy().dropna(subset=['Last_Contact_Date', 'Recent_Annualized_Giving'])
    
    if not rg_df.empty:
        if rg_gran == "Month":
            rg_df['Period'] = rg_df['Last_Contact_Date'].dt.to_period('M').astype(str)
        else:
            rg_df['Period'] = rg_df['Last_Contact_Date'].dt.to_period('Q').astype(str)
            
        agg_map_rg = {"Average": "mean", "Sum": "sum", "Median": "median"}
        rg_grouped = rg_df.groupby("Period")["Recent_Annualized_Giving"].agg(agg_map_rg[rg_agg]).reset_index()
        
        fig_rg = px.line(
            rg_grouped,
            x="Period",
            y="Recent_Annualized_Giving",
            markers=True,
            title=f"Recent Giving Trend by Last Contact ({rg_gran})",
            color_discrete_sequence=["#C85A54"], # Rust
            template="plotly_white"
        )
        fig_rg.update_layout(hovermode="x unified")
        st.plotly_chart(fig_rg, use_container_width=True)
    else:
        st.write("No data available with Last Contact Date.")

