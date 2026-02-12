import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from xhtml2pdf import pisa
import io
import textwrap
import re

# --- Configuration ---
st.set_page_config(
    page_title="CPO Report",
    page_icon="📄",
    layout="wide"
)

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
            'Churn_Risk_Score'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Ensure string columns are strings
        str_cols = ['Geography', 'Industry', 'Relationship_Stage', 'Churn_Risk_Category', 'Drift_Status', 'Assigned_RM']
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '')

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# --- Metric Calculations ---

# 1. Ghost Economy
ghost_df = df[
    (df['Lifetime_Giving'] > 500000) & 
    (df['Giving_Last_24_Months'] == 0)
].sort_values(by="Lifetime_Giving", ascending=False)
ghost_total_ltv = ghost_df['Lifetime_Giving'].sum()
ghost_count = len(ghost_df)

# 2. Resource Misallocation (Alice vs Ben)
rm_stats = df.groupby('Assigned_RM').agg({
    'Touchpoints_Last_12_Months': 'sum',
    'Giving_Last_24_Months': 'sum',
    'Donor_ID': 'count'
}).reset_index()
rm_stats['Revenue_Per_Touchpoint'] = rm_stats['Giving_Last_24_Months'] / rm_stats['Touchpoints_Last_12_Months'].replace(0, 1)

alice_stats = rm_stats[rm_stats['Assigned_RM'] == 'Alice'].iloc[0] if 'Alice' in rm_stats['Assigned_RM'].values else None
ben_stats = rm_stats[rm_stats['Assigned_RM'] == 'Ben'].iloc[0] if 'Ben' in rm_stats['Assigned_RM'].values else None

# 3. Data Hygiene
missing_contact_count = df['Last_Contact_Date'].isna().sum()
total_donors = len(df)
data_gaps_pct = (missing_contact_count / total_donors) * 100

# 4. Burnout Risk (Alice)
alice_touchpoints = alice_stats['Touchpoints_Last_12_Months'] if alice_stats is not None else 0
avg_touchpoints = rm_stats['Touchpoints_Last_12_Months'].mean()

# --- HELPER FUNCTIONS ---

def format_markdown_to_html(text):
    """Converts basic Markdown bold/bullets/headers to HTML for PDF generation."""
    # Bold: **text** -> <strong>text</strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Headers: ### Text -> <h3>Text</h3>
    lines = text.split('\n')
    html_lines = []
    in_list = False
    
    for line in lines:
        stripped = line.strip()
        
        # Lists
        if stripped.startswith('* '):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            content = stripped[2:]
            html_lines.append(f"<li>{content}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            
            # Headers
            if stripped.startswith('### '):
                html_lines.append(f"<h3>{stripped[4:]}</h3>")
            elif stripped.startswith('## '):
                html_lines.append(f"<h2>{stripped[3:]}</h2>")
            elif stripped:
                html_lines.append(f"{stripped}<br>")
    
    if in_list:
        html_lines.append("</ul>")
        
    return "".join(html_lines)

# --- CONTENT VARIABLES ---

# Header Info
report_to = "Chief Partnerships Officer, ABC Philanthropy"
report_from = "Head of Development Operations (Candidate)"
report_subject = "Operational Diagnosis and Optimization Plan"
report_date = datetime.now().strftime('%B %d, %Y')

# 1. Executive Summary
exec_sum_html = f"""
ABC Philanthropy is currently operating under a notable <strong>Resource Misallocation.</strong> 
While the fundraising team reports high activity levels, this effort is structurally misaligned with revenue potential. 
The organization is over-servicing low-yield "time sink" accounts while neglecting a massive 
<strong>"shadow endowment"</strong> of dormant, high-capacity donors.
<br><br>
Our forensic audit of the Task 1 dataset confirms that <span class="highlight">${ghost_total_ltv/1000000:.2f} million</span> in Lifetime Value (LTV) 
is currently trapped in "Ghost" accounts: Donors who have given significantly in the past but have contributed $0 in the last 24 months. 
Reactivating even a fraction of this capital offers a significantly higher ROI than current acquisition efforts.
<br><br>
Simultaneously, the "burnout" signal detected by leadership is real but unevenly distributed. 
It stems from a "Friend-Raising" operational model where Relationship Managers (RMs) like "Alice" generate 
high activity but deliver lower yields, largely due to a lack of data-driven prioritization.
<br><br>
This report outlines a strategy to pivot from <strong>Activity-Based Fundraising</strong> to 
<strong>Velocity-Based Stewardship</strong>, leveraging the "Drift" and "Engagement" metrics buried in our 
current data to automate prioritization and unlock millions in near-term revenue.
"""

# 2. Ghost Economy
insight_1_intro = f"""
### **The Diagnostic**
We identified a specific cohort of **"Ghost" donors**: individuals with **LTV > 500k USD** who have contributed **$0** in the last 24 months. 
These are not cold leads; they are lapsed partners who have already proven their seven-figure capacity.
"""

insight_1_root_cause = f"""
### **Root Cause Analysis**
1. **The "Charlie" Bottleneck:** Relationship Manager "Charlie" effectively controls the organization's legacy endowment but has the lowest activity rate. This suggests that by assigning to a senior RM who is not actively working them, these high-value donors are being "warehoused" rather than stewarded.
2. **Systemic Amnesia (D007):** Donor D007 ($2.1M) is explicitly noted as dormant "after staff turnover." This indicates a failure of **Institutional Memory**. The CRM lacks a "Transition Workflow" to ensure relationships survive staff departures.
3. **Data Hygiene Gaps:** **{missing_contact_count} donors** are penalized with a "High Risk" score simply because their Last_Contact_Date is null. The system defaults these to 999+ days, rendering them invisible to "Recent Activity" reports.
"""

insight_1_reco = """
### **Strategic Recommendation**
**Launch "Operation Reconnect" (Next 30 Days):**
* **Action:** Reassign the Top 5 Ghost Donors (e.g., D007, D012) to a "Reactivation Task Force" led by the CPO or the most efficient RM (Ben).
* **The Pitch:** "We dropped the ball during our transition, but we want to show you what your past contributions have achieved."
* **Goal:** Secure one re-engagement meeting per donor. A 20% success rate here unlocks ~$2M in immediate pipeline.
"""

# 3. Asymmetric Efficiency
insight_2_intro = """
The CPO’s concern about burnout is validated by the data, but it is not a team-wide issue. 
It is a structural inefficiency concentrated in the "High Volume" portfolio model.
"""

insight_2_detail = f"""
### **The "Alice Trap": Activity ≠ Achievement**
Alice is deploying **{alice_touchpoints} touchpoints** (significantly higher than average) but generating lower revenue yield. 
The data reveals she is trapped in "Activity Loops" with donors who consume time but do not advance.

* **Example (D002):** Alice logged **11 touchpoints** (nearly monthly) for a donor who gave only **$25k**. Notes: *"Very responsive, lots of calls; hasn’t converted beyond small gifts."*
* **Example (D023):** 9 touchpoints for $45k. Notes: *"Attends events; slow to commit."*

Alice is effectively providing "Concierge Stewardship" to mid-tier donors. This is unsustainable. 
If ABC Philanthropy doubles its donor base, Alice cannot double her calls.
"""

insight_2_reco = """
### **Strategic Recommendation**
**Implement an "Efficiency Floor" (Next Quarter):**
* **Policy:** Donors with LTV < $50k are capped at **2 bespoke interactions per year**.
* **Substitution:** Move these donors to a "Digital Stewardship" track (automated impact reports, webinar invites) to free up Alice's capacity.
* **Redirection:** Direct Alice’s freed capacity toward **Ghost Donors** or **Accelerating Donors**, where the ROI on her high-touch style is significantly higher.
"""

# 4. Roadmap
roadmap_text = """
To sustain growth without burnout, we must transition from "Reactive Data Entry" to "Proactive Intelligence."

**Phase 1: The "Data Triage" Sprint (Weeks 1-4)**
* **Objective:** Eliminate "False High Risk" signals.
* **Action:** The Ops team will manually verify and input Last_Contact_Date for the Top 20 LTV donors, specifically targeting the "Nulls".
* **Outcome:** A clean "Churn Risk" dashboard that alerts RMs to *real* problems, not data entry gaps.

**Phase 2: The "Shadow Ledger" Implementation (Months 2-3)**
* **Objective:** Align CRM metrics with ABC Philanthropy's "Active Grantmaking" mission.
* **Context:** ABC Philanthropy advises donors to give to external funds. Current "Revenue" fields likely miss this "Money Moved."
* **Action:** Create a custom object/field for **"Influenced Capital"** distinct from **"Operating Revenue."**
* **Why:** This validates the work of RMs who may be moving millions to the *cause* without moving it through the *org*.

**Phase 3: Automated "Drift" Dashboards (Quarter 2)**
* **Objective:** Prevent the next D007 (the $2.1M loss).
* **Action:** Deploy the **Drift Ratio** metric to the RM dashboard.
* **Workflow:** When a donor's Drift Ratio drops below **0.8** (indicating a 20% deceleration), an automated alert is sent to the RM. This shifts the team from *autopsy* to *preventative care*.
"""

conclusion_text = f"""
### **Conclusion**
The data proves that ABC Philanthropy does not need **more** leads to grow; it needs **better** allocation of the leads it already has. 
By reactivating the **${ghost_total_ltv/1000000:.1f}M Ghost Economy** and implementing an **Efficiency Floor** for high-touch RMs, 
we can achieve the 12-24 month growth targets while actively protecting the team from the burnout of low-yield churn.
"""

# --- PDF GENERATOR ---
def generate_pdf_html():
    css = """
    <style>
        @page { size: letter; margin: 2cm; }
        body { font-family: 'Helvetica', sans-serif; font-size: 10pt; color: #333; line-height: 1.5; }
        h1 { color: #1A5F57; font-size: 18pt; border-bottom: 2px solid #1A5F57; padding-bottom: 10px; margin-bottom: 20px; }
        h2 { color: #1A5F57; font-size: 14pt; margin-top: 25px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
        h3 { color: #2D9B8F; font-size: 12pt; margin-top: 15px; margin-bottom: 5px; }
        .highlight { color: #C85A54; font-weight: bold; }
        .box { background-color: #F8F6F3; padding: 15px; border: 1px solid #D4A574; margin: 15px 0; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 20px; }
        th, td { border: 1px solid #DDD; padding: 4px; text-align: left; }
        th { background-color: #F0F0F0; color: #1A5F57; font-weight: bold; }
        ul { margin-bottom: 10px; }
        li { margin-bottom: 5px; }
    </style>
    """
    
    # Tables for PDF
    ghost_rows = ''.join([f"<tr><td>{row['Donor_ID']}</td><td>{row['Industry']}</td><td>${row['Lifetime_Giving']:,.0f}</td><td>{row['Assigned_RM']}</td><td>{row['Drift_Status']}</td></tr>" for index, row in ghost_df.head(5).iterrows()])
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>{css}</head>
    <body>
        <h1>{report_subject}</h1>
        <p><strong>To:</strong> {report_to}<br/>
        <strong>From:</strong> {report_from}<br/>
        <strong>Date:</strong> {report_date}</p>

        <div class="box">
            <strong>1. Executive Summary</strong><br/>
            {exec_sum_html}
        </div>

        <h2>2. The "Ghost Economy": A ${ghost_total_ltv/1000000:.1f} Million Opportunity</h2>
        <p>{format_markdown_to_html(insight_1_intro)}</p>
        
        <h3>Table 1: The "Ghost" Segment (Top 5 by LTV)</h3>
        <table>
            <thead><tr><th>Donor ID</th><th>Industry</th><th>Lifetime Giving</th><th>Assigned RM</th><th>Status</th></tr></thead>
            <tbody>{ghost_rows}</tbody>
        </table>

        <p>{format_markdown_to_html(insight_1_root_cause)}</p>
        <p>{format_markdown_to_html(insight_1_reco)}</p>

        <h2>3. Asymmetric Efficiency: Solving the Burnout Crisis</h2>
        <p>{format_markdown_to_html(insight_2_intro)}</p>
        
        <h3>The Efficiency Matrix</h3>
        <table style="width:70%">
            <thead><tr><th>Metric</th><th>Alice (High Volume)</th><th>Ben (High Efficiency)</th></tr></thead>
            <tbody>
                <tr><td>Activity (Touchpoints 12M)</td><td>{alice_stats['Touchpoints_Last_12_Months']:,.0f}</td><td>{ben_stats['Touchpoints_Last_12_Months']:,.0f}</td></tr>
                <tr><td>Revenue (Last 24M)</td><td>${alice_stats['Giving_Last_24_Months']:,.0f}</td><td>${ben_stats['Giving_Last_24_Months']:,.0f}</td></tr>
                <tr><td>Efficiency (Rev/Touch)</td><td><span class="highlight">${alice_stats['Revenue_Per_Touchpoint']:,.0f}</span></td><td><strong>${ben_stats['Revenue_Per_Touchpoint']:,.0f}</strong></td></tr>
            </tbody>
        </table>

        <p>{format_markdown_to_html(insight_2_detail)}</p>
        <p>{format_markdown_to_html(insight_2_reco)}</p>

        <h2>4. Structural Roadmap: Building the Intelligence Engine</h2>
        <p>{format_markdown_to_html(roadmap_text)}</p>
        
        <p>{format_markdown_to_html(conclusion_text)}</p>
    </body>
    </html>
    """
    return html

# --- PDF Generation Function ---
def create_pdf(html_content):
    pdf_file = io.BytesIO()
    pisa_status = pisa.CreatePDF(src=html_content, dest=pdf_file)
    if pisa_status.err: return None
    return pdf_file.getvalue()

# Pre-generate PDF for Sidebar
pdf_bytes = create_pdf(generate_pdf_html())

# --- Sidebar ---
with st.sidebar:
    st.header("Report Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="💾 CSV Data", data=csv, file_name='task_1_processed_v2.csv', mime='text/csv')
    with col2:
        if pdf_bytes:
            st.download_button(label="⬇️ PDF Report", data=pdf_bytes, file_name=f"ABC_Philanthropy_Audit.pdf", mime="application/pdf")
    
    st.info("Download the full processed dataset or the PDF compliance report.")

# --- WEB VIEW (Custom Styled) ---

# Custom CSS for nicer aesthetics
st.markdown("""
<style>
    .report-title { font-family: 'Merriweather', serif; color: #1A5F57; font-size: 2.5rem; border-bottom: 3px solid #1A5F57; padding-bottom: 10px; margin-bottom: 20px; }
    .report-meta { color: #666; font-size: 1rem; margin-bottom: 30px; }
    .exec-summary-box { background-color: #F8F6F3; border-left: 5px solid #D4A574; padding: 20px; margin: 20px 0; border-radius: 4px; color: #333; font-size: 1.1rem; line-height: 1.6; }
    .section-header { color: #1A5F57; font-size: 1.8rem; margin-top: 40px; margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
    .highlight-red { color: #C85A54; font-weight: bold; }
    .highlight { color: #C85A54; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #1A5F57; }
    /* Fix for lists in markdown to match report style */
    ul { margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="report-title">{report_subject}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="report-meta">To: {report_to} <br> From: {report_from} <br> Date: {report_date}</div>', unsafe_allow_html=True)

# 1. Executive Summary
st.markdown(f'<div class="exec-summary-box"><strong>1. Executive Summary</strong><br>{exec_sum_html}</div>', unsafe_allow_html=True)

# 2. Ghost Economy
st.markdown(f'<div class="section-header">2. The "Ghost Economy": A ${ghost_total_ltv/1000000:.1f} Million Opportunity</div>', unsafe_allow_html=True)
st.markdown(insight_1_intro)

st.markdown("##### Table 1: The 'Ghost' Segment (Top 5 by LTV)")
st.dataframe(
    ghost_df[['Donor_ID', 'Industry', 'Lifetime_Giving', 'Assigned_RM', 'Drift_Status']].head(5),
    hide_index=True,
    column_config={"Lifetime_Giving": st.column_config.NumberColumn("LTV", format="$%d")}
)

st.markdown(insight_1_root_cause)
st.markdown(insight_1_reco)

# 3. Asymmetric Efficiency
st.markdown('<div class="section-header">3. Asymmetric Efficiency: Solving the Burnout Crisis</div>', unsafe_allow_html=True)
st.markdown(insight_2_intro)

st.markdown("##### The Efficiency Matrix")
col1, col2 = st.columns(2)
with col1:
    st.dataframe(pd.DataFrame({
        "Metric": ["Activity (Touchpoints 12M)", "Revenue (Last 24M)", "Efficiency (Rev/Touch)"],
        "Alice (High Volume)": [f"{alice_stats['Touchpoints_Last_12_Months']:,.0f}", f"${alice_stats['Giving_Last_24_Months']:,.0f}", f"${alice_stats['Revenue_Per_Touchpoint']:,.0f}"],
        "Ben (High Efficiency)": [f"{ben_stats['Touchpoints_Last_12_Months']:,.0f}", f"${ben_stats['Giving_Last_24_Months']:,.0f}", f"${ben_stats['Revenue_Per_Touchpoint']:,.0f}"]
    }), hide_index=True)

st.markdown(insight_2_detail)
st.markdown(insight_2_reco)

# 4. Roadmap
st.markdown('<div class="section-header">4. Structural Roadmap: Building the Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown(roadmap_text)

# Conclusion
st.markdown("---")
st.markdown(conclusion_text)
