import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- Load the raw data once to avoid reloading on every interaction ---
@st.cache_data
def load_data():
    try:
        raw_data = pd.read_csv('Clean_Sample.csv')
        return raw_data
    except FileNotFoundError:
        st.error("Error: 'Clean_Sample.csv' not found. Please upload the file to run the app.")
        return pd.DataFrame()

raw_data = load_data()
if raw_data.empty:
    st.stop()

# --- AGENTIC AI TOOLS (Your previously developed functions) ---
@st.cache_resource
def run_segmentation_model(df):
    st.info("Running Customer Segmentation Model...")
    features = ['Sold-to', 'Order quantity in Base Unit', 'Order Value']
    df_features = df[features]
    customer_df = df_features.groupby('Sold-to').agg(
        total_order_quantity=('Order quantity in Base Unit', 'sum'),
        total_order_value=('Order Value', 'sum')
    ).reset_index()
    customer_ids = customer_df['Sold-to']
    customer_features = customer_df.drop('Sold-to', axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_features)
    scaled_df = pd.DataFrame(scaled_features, columns=customer_features.columns)
    optimal_k = 4
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init=10)
    final_kmeans.fit(scaled_df)
    customer_df['Cluster'] = final_kmeans.labels_
    customer_df['Sold-to'] = customer_ids
    st.success("Customer Segmentation Model complete.")
    return customer_df

@st.cache_resource
def run_market_basket_analysis(df, customer_clusters):
    st.info("Running Product Association Model...")
    df_with_clusters = pd.merge(df, customer_clusters[['Sold-to', 'Cluster']], on='Sold-to', how='left')
    def get_cluster_rules(cluster_id, min_support_val):
        df_cluster = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        transactions_cluster = df_cluster.groupby('Order #')['Material'].apply(list).reset_index(name='Items')
        if transactions_cluster.shape[0] < 2: return pd.DataFrame()
        te = TransactionEncoder()
        try:
            te_ary = te.fit(transactions_cluster['Items']).transform(transactions_cluster['Items'])
            df_encoded_cluster = pd.DataFrame(te_ary, columns=te.columns_)
        except ValueError: return pd.DataFrame()
        frequent_itemsets_cluster = apriori(df_encoded_cluster, min_support=min_support_val, use_colnames=True)
        if frequent_itemsets_cluster.empty: return pd.DataFrame()
        rules_cluster = association_rules(frequent_itemsets_cluster, metric="lift", min_lift=1)
        rules_cluster['Cluster'] = cluster_id
        return rules_cluster.sort_values(['confidence', 'lift'], ascending=[False, False])

    rules_cluster_0 = get_cluster_rules(0, min_support_val=0.01)
    rules_cluster_1 = get_cluster_rules(1, min_support_val=0.5)
    rules_cluster_2 = get_cluster_rules(2, min_support_val=0.5)
    rules_cluster_3 = get_cluster_rules(3, min_support_val=0.5)
    all_rules = pd.concat([rules_cluster_0, rules_cluster_1, rules_cluster_2, rules_cluster_3])
    all_rules['antecedents'] = all_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    all_rules['consequents'] = all_rules['consequents'].apply(lambda x: ', '.join(list(x)))
    st.success("Product Association Model complete.")
    return all_rules

def find_next_best_action(customer_identifier, customer_clusters_df, combined_rules_df):
    customer_data = customer_clusters_df[customer_clusters_df['Sold-to'] == customer_identifier]
    if customer_data.empty:
        return f"Error: Customer ID {customer_identifier} not found."
    customer_segment = customer_data['Cluster'].iloc[0]
    segment_rules = combined_rules_df[combined_rules_df['Cluster'] == customer_segment]
    if segment_rules.empty:
        return f"No specific action found for customer {customer_identifier}. Recommendation: Personalized sales approach."
    top_rule = segment_rules.sort_values(['confidence', 'lift'], ascending=False).iloc[0]
    antecedent = top_rule['antecedents']
    consequent = top_rule['consequents']
    confidence = top_rule['confidence'] * 100
    recommendation = (
        f"Next Best Action for customer {customer_identifier} (Cluster {customer_segment}):\n"
        f"  - Based on the purchase of: {antecedent}\n"
        f"  - Recommend product(s): {consequent}\n"
        f"  - Confidence: {confidence:.2f}%"
    )
    return recommendation

def model_monitor(combined_rules_df):
    LIFT_THRESHOLD = 2.0
    NUM_RULES_THRESHOLD = 50
    current_avg_lift = combined_rules_df['lift'].mean()
    current_num_rules = len(combined_rules_df)
    if current_avg_lift < LIFT_THRESHOLD or current_num_rules < NUM_RULES_THRESHOLD:
        status = "⚠️ WARNING: Models may need to be re-trained."
        color = "red"
    else:
        status = "✅ All Models are healthy."
        color = "green"
    return status, color

# --- AGENTIC AI ORCHESTRATOR FOR THE UI ---
st.title("Enterprise AI Assistant")
st.write("A self-monitoring system for customer segmentation and strategic recommendations.")

# Run the full pipeline
customer_clusters_output = run_segmentation_model(raw_data)
rules_output = run_market_basket_analysis(raw_data, customer_clusters_output)

# --- UI PART 1: MODEL MONITORING ---
st.header("1. System Health Check")
status_message, status_color = model_monitor(rules_output)
if status_color == "green":
    st.success(status_message)
else:
    st.warning(status_message)

# --- UI PART 2: NEXT BEST ACTION ---
st.header("2. Next Best Action Finder")
st.write("Get a specific, data-driven recommendation for any customer.")
customer_id_list = customer_clusters_output['Sold-to'].unique()
customer_id_input = st.selectbox("Select a Customer ID:", options=customer_id_list)
if st.button("Find Action"):
    nba_result = find_next_best_action(customer_id_input, customer_clusters_output, rules_output)
    st.text(nba_result)

# --- UI PART 3: WHAT-IF ANALYSIS ---
st.header("3. What-If Campaign Simulator")
st.write("Simulate the financial impact of moving customers to a high-value cluster.")
num_customers_to_move = st.slider(
    "Number of Customers to Move from Cluster 0 to Cluster 1:",
    min_value=0, max_value=1000, value=0, step=10
)
current_total_value = customer_clusters_output['total_order_value'].sum()
avg_val_high = customer_clusters_output[customer_clusters_output['Cluster'] == 1]['total_order_value'].mean()
avg_val_low = customer_clusters_output[customer_clusters_output['Cluster'] == 0]['total_order_value'].mean()
value_increase_per_customer = avg_val_high - avg_val_low
simulated_value = current_total_value + (num_customers_to_move * value_increase_per_customer)
col1, col2 = st.columns(2)
col1.metric("Current Total Order Value", f"${current_total_value:,.0f}")
col2.metric(
    "Simulated Total Order Value",
    f"${simulated_value:,.0f}",
    delta=f"Increase: ${simulated_value - current_total_value:,.0f}"
)