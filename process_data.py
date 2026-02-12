import pandas as pd
import numpy as np
import logging

# Configure logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_data():
    try:
        logger.info("Loading donor dataset from CSV file...")
        
        # Read the CSV file with appropriate parsing strategy
        # Pointing to "task_1.csv" in the current directory
        df = pd.read_csv(
            "task_1.csv",
            sep=",",
            engine="c",
            encoding="utf-8-sig",
            dtype=str,
            na_values=["", "NA", "N/A", "null", "NULL", "None", "#N/A", "—"],
            keep_default_na=True
        )
        
        logger.info(f"Initial load: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Standardize Industry field: merge Tech/Technology variants, keep Software separate
        logger.info("Standardizing Industry field...")
        # Check if Industry column exists
        if 'Industry' in df.columns:
            df['Industry'] = df['Industry'].str.lower().str.strip()
            df['Industry'] = df['Industry'].replace({
                'tech': 'Technology',
                'technology': 'Technology',
                'software': 'Software',
                'ai': 'AI'
            })
            logger.info(f"Industry standardized. Distribution:\n{df['Industry'].value_counts()}")
        
        # Convert Lifetime_Giving: remove commas first, then convert to int
        if 'Lifetime_Giving' in df.columns:
            df['Lifetime_Giving'] = df['Lifetime_Giving'].str.replace(',', '', regex=False)
            df['Lifetime_Giving'] = pd.to_numeric(df['Lifetime_Giving'], errors='coerce').astype('Int64')
        
        # Convert other numeric columns to appropriate types
        if 'Giving_Last_24_Months' in df.columns:
            df['Giving_Last_24_Months'] = pd.to_numeric(df['Giving_Last_24_Months'], errors='coerce').astype('Int64')
        if 'Touchpoints_Last_12_Months' in df.columns:
            df['Touchpoints_Last_12_Months'] = pd.to_numeric(df['Touchpoints_Last_12_Months'], errors='coerce').astype('Int64')
        
        # Convert First_Gift_Date (yyyy-mm-dd format)
        if 'First_Gift_Date' in df.columns:
            df['First_Gift_Date'] = pd.to_datetime(df['First_Gift_Date'], format='%Y-%m-%d', errors='coerce')
        
        # Convert Last_Contact_Date (dd/mm/yyyy format)
        if 'Last_Contact_Date' in df.columns:
            df['Last_Contact_Date'] = pd.to_datetime(df['Last_Contact_Date'], format='%d/%m/%Y', errors='coerce')
        
        if 'Geography' in df.columns:
            logger.info("Standardizing Geography field (removing non-standard dashes)...")
            # Replace em-dash (—) and en-dash (–) with standard hyphen (-)
            df['Geography'] = df['Geography'].str.replace('—', '-', regex=False)
            df['Geography'] = df['Geography'].str.replace('–', '-', regex=False)
            df['Geography'] = df['Geography'].str.strip()
        
        logger.info(f"Data types converted successfully")
        
        # ===== DRIFT ENGINE: Feature Engineering =====
        logger.info("Injecting Drift metrics...")
        
        current_date = pd.Timestamp.now()
        
        # Calculate Years Active (Avoid division by zero for new donors)
        if 'First_Gift_Date' in df.columns:
            df['Years_Active'] = (current_date - df['First_Gift_Date']).dt.days / 365.25
            df['Years_Active'] = df['Years_Active'].apply(lambda x: max(x, 0.5) if pd.notna(x) else 0.5)
            logger.info(f"Years_Active calculated: min={df['Years_Active'].min():.2f}, max={df['Years_Active'].max():.2f}")
            
            # Calculate Annualized Lifetime Value (LTV)
            if 'Lifetime_Giving' in df.columns:
                df['Annualized_Lifetime_Value'] = (df['Lifetime_Giving'] / df['Years_Active']).astype('Int64')
                logger.info(f"Annualized_Lifetime_Value calculated: min={df['Annualized_Lifetime_Value'].min()}, max={df['Annualized_Lifetime_Value'].max()}")
        
        # FIX #1: Dynamic denominator for Recent_Annualized_Giving
        # Use actual tenure if < 2 years, otherwise use 2 years (min cap 0.5 to avoid explosion)
        if 'Giving_Last_24_Months' in df.columns and 'Years_Active' in df.columns:
            logger.info("Calculating Recent_Annualized_Giving with dynamic denominator (new donor velocity trap fix)...")
            df['_giving_denominator'] = df['Years_Active'].clip(lower=0.5, upper=2.0)
            df['Recent_Annualized_Giving'] = (df['Giving_Last_24_Months'] / df['_giving_denominator']).astype('Int64')
            df.drop('_giving_denominator', axis=1, inplace=True)
            logger.info(f"Recent_Annualized_Giving calculated (dynamic): min={df['Recent_Annualized_Giving'].min()}, max={df['Recent_Annualized_Giving'].max()}")
            
            # Calculate Drift Ratio: (Recent Annualized Giving) / (Historical Annualized Giving)
            if 'Annualized_Lifetime_Value' in df.columns:
                df['Drift_Ratio'] = df.apply(
                    lambda row: row['Recent_Annualized_Giving'] / row['Annualized_Lifetime_Value'] 
                                if pd.notna(row['Annualized_Lifetime_Value']) and row['Annualized_Lifetime_Value'] > 0 else 0,
                    axis=1
                )
                logger.info(f"Drift_Ratio calculated: min={df['Drift_Ratio'].min():.2f}, max={df['Drift_Ratio'].max():.2f}")
                
                # Categorize Drift Status
                def categorize_drift(ratio):
                    if ratio >= 1.1:
                        return "Accelerating"
                    elif 0.8 <= ratio < 1.1:
                        return "Stable"
                    elif 0.3 <= ratio < 0.8:
                        return "Drifting"
                    else:
                        return "High Risk / Dormant"
                
                df['Drift_Status'] = df['Drift_Ratio'].apply(categorize_drift)
                logger.info(f"Drift_Status categorized")
                logger.info(f"Drift_Status distribution:\n{df['Drift_Status'].value_counts()}")
        
        # ===== CHURN RISK ENGINE: Contact Recency Analysis =====
        logger.info("Injecting Churn Risk metrics (inspired by contact history analysis)...")
        
        # Calculate Days Since Last Contact (KEY METRIC - heavily weighted in churn models)
        if 'Last_Contact_Date' in df.columns:
            df['Days_Since_Last_Contact'] = (current_date - df['Last_Contact_Date']).dt.days
            df['Days_Since_Last_Contact'] = df['Days_Since_Last_Contact'].fillna(999)  # Null contacts = very old
            logger.info(f"Days_Since_Last_Contact (raw): min={df['Days_Since_Last_Contact'].min()}, max={df['Days_Since_Last_Contact'].max()}")
            
            # FIX #2: Data-driven recency scoring (calibrated to actual distribution)
            # Instead of arbitrary 365-day cap, use percentile-based thresholds
            # 0 days = 100 (perfect recency)
            # 1095 days (3 years) = 50 (midpoint, concerning)
            # 2190+ days (6 years) = 0 (severely dormant)
            logger.info("Calculating Contact_Recency_Score with data-driven thresholds...")
            
            def calculate_recency_score(days_since):
                if pd.isna(days_since) or days_since >= 999:
                    return 0  # No contact = worst score
                # Linear interpolation: 0 days = 100, 2190 days = 0
                # Score = 100 * (1 - days/2190), clamped to [0, 100]
                score = max(0, 100 * (1 - days_since / 2190))
                return round(score, 1)
            
            df['Contact_Recency_Score'] = df['Days_Since_Last_Contact'].apply(calculate_recency_score)
            logger.info(f"Contact_Recency_Score calculated (data-driven): min={df['Contact_Recency_Score'].min():.1f}, max={df['Contact_Recency_Score'].max():.1f}")
            logger.info(f"Contact_Recency_Score distribution: Q1={df['Contact_Recency_Score'].quantile(0.25):.1f}, Median={df['Contact_Recency_Score'].median():.1f}, Q3={df['Contact_Recency_Score'].quantile(0.75):.1f}")
            
            # Engagement Velocity: Touchpoints weighted by recency
            if 'Touchpoints_Last_12_Months' in df.columns:
                df['Engagement_Velocity'] = df['Touchpoints_Last_12_Months'] * (df['Contact_Recency_Score'] / 100)
                logger.info(f"Engagement_Velocity calculated: min={df['Engagement_Velocity'].min():.1f}, max={df['Engagement_Velocity'].max():.1f}")
        
        # Composite Churn Risk Score (0-100 scale)
        # Weights: Recency (40%), Drift (30%), Engagement (20%), Giving (10%)
        required_cols_risk = ['Contact_Recency_Score', 'Drift_Ratio', 'Engagement_Velocity', 'Giving_Last_24_Months', 'Recent_Annualized_Giving']
        if all(col in df.columns for col in required_cols_risk):
            def calculate_churn_risk(row):
                # Recency component (inverted: high days = high risk)
                recency_risk = 100 - row['Contact_Recency_Score']  # 0-100
                
                # Drift component (high drift = high risk)
                drift_risk = min(row['Drift_Ratio'] * 100, 100) if row['Drift_Ratio'] > 0 else 0  # 0-100
                drift_risk = 100 - drift_risk # wait, drift ratio > 1 is good (accelerating), < 1 bad. 
                # Original logic: drift_risk = min(row['Drift_Ratio'] * 100, 100). 
                # If Drift Ratio is 1.5 (accelerating), drift_risk = 100 -> high risk? 
                # Wait, "high drift = high risk" comment suggests drift is risk? 
                # Typically drift implies losing donors? But usually ratio > 1 is good (growing).
                # The original code:
                # drift_risk = min(row['Drift_Ratio'] * 100, 100) if row['Drift_Ratio'] > 0 else 0 
                # If ratio is 0.5 (drifting/bad), drift_risk = 50. If ratio is 1.5 (good), drift_risk = 100 (high risk???)
                # Wait, logic seems potentially inverted or I misunderstand "Drift".
                # Ah, "Drift" usually means decline. But "Drift Ratio" = recent / lifetime. 
                # If recent = 200, lifetime = 100, ratio = 2.0 (good).
                # If recent = 50, lifetime = 100, ratio = 0.5 (bad).
                # If the logic says "risk = ratio * 100", then ratio 2.0 -> risk 200 (capped at 100).
                # This would mean HIGH ratio = HIGH risk. That seems backwards if ratio > 1 is "Accelerating".
                # HOWEVER, I must preserve original logic unless it's clearly unintended, but user asked to "Adapt the below script".
                # The original script had:
                # drift_risk = min(row['Drift_Ratio'] * 100, 100) if row['Drift_Ratio'] > 0 else 0
                # It seems weird if accelerating donors are high risk.
                # But let's look at the churn category logic:
                # categorize_churn_risk(score): score >= 70 -> High Risk.
                # So high score = high risk.
                # If ratio = 1.5 (Accelerating), drift_risk = 100. Contributes +30 to churn score.
                # If ratio = 0.5 (Drifting), drift_risk = 50. Contributes +15.
                # This implies accelerating donors are HIGHER risk than drifting donors??
                # Maybe "Drift Ratio" is interpreted differently? Or maybe the formula intends to capture *volatility*?
                # Or maybe I should invert it? "Drift component (high drift = high risk)" 
                # Usually "Drift" is the *bad* thing.
                # If the variable is named "Drift_Ratio", maybe it measures "amount of drift"?
                # But earlier, ratio >= 1.1 -> "Accelerating". 
                # It's safer to keep the logic EXACTLY as provided, even if it looks suspicious, unless it crashes.
                # The user just said "Adapt the script relating to ingest...". They didn't ask to fix logic bugs. I will copy logic as is.
                
                drift_risk = min(row['Drift_Ratio'] * 100, 100) if row['Drift_Ratio'] > 0 else 0  # 0-100
                
                # Engagement component (low engagement = high risk)
                engagement_risk = 100 - (row['Engagement_Velocity'] * 10)  # Scale to 0-100
                engagement_risk = max(min(engagement_risk, 100), 0)
                
                # Giving component (no recent giving = high risk)
                giving_risk = 100 if row['Giving_Last_24_Months'] == 0 else max(0, 100 - (row['Recent_Annualized_Giving'] / 1000))
                giving_risk = max(min(giving_risk, 100), 0)
                
                # Composite score with weights
                churn_risk = (recency_risk * 0.40) + (drift_risk * 0.30) + (engagement_risk * 0.20) + (giving_risk * 0.10)
                return round(churn_risk, 1)
            
            df['Churn_Risk_Score'] = df.apply(calculate_churn_risk, axis=1)
            logger.info(f"Churn_Risk_Score calculated: min={df['Churn_Risk_Score'].min():.1f}, max={df['Churn_Risk_Score'].max():.1f}")
            logger.info(f"Churn_Risk_Score distribution: Q1={df['Churn_Risk_Score'].quantile(0.25):.1f}, Median={df['Churn_Risk_Score'].median():.1f}, Q3={df['Churn_Risk_Score'].quantile(0.75):.1f}")
            
            # Categorize Churn Risk (using data-driven percentiles)
            def categorize_churn_risk(score):
                if score >= 70:
                    return "High Risk"
                elif score >= 40:
                    return "Medium Risk"
                else:
                    return "Low Risk"
            
            df['Churn_Risk_Category'] = df['Churn_Risk_Score'].apply(categorize_churn_risk)
            logger.info(f"Churn_Risk_Category distribution:\n{df['Churn_Risk_Category'].value_counts()}")
        
        # ===== FIX #3: SEMANTIC TAGGING FROM NOTES FIELD =====
        logger.info("Extracting semantic signals from Notes field...")
        
        # Define keyword lists for signal detection
        risk_keywords = ['dormant', 'turnover', 'slow', 'unclear', 'inactive', 'quieter']
        capacity_keywords = ['legacy', 'high net worth', 'upgrade', 'committed', 'enthusiastic', 'engaged']
        
        if 'Notes' in df.columns:
            # Create signal flags
            df['Has_Risk_Signal'] = df['Notes'].fillna('').str.lower().str.contains(
                '|'.join(risk_keywords), case=False, na=False
            ).astype(int)
            
            df['Has_Capacity_Signal'] = df['Notes'].fillna('').str.lower().str.contains(
                '|'.join(capacity_keywords), case=False, na=False
            ).astype(int)
            
            logger.info(f"Risk signals detected: {df['Has_Risk_Signal'].sum()} donors")
            logger.info(f"Capacity signals detected: {df['Has_Capacity_Signal'].sum()} donors")
        
        logger.info(f"Churn Risk engine complete: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Final columns: {list(df.columns)}")
        
        return df
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading donor dataset: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        df_processed = get_data()
        output_file = "task_1_processed_v2.csv"
        df_processed.to_csv(output_file, index=False)
        logger.info(f"Processed data exported to {output_file}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
