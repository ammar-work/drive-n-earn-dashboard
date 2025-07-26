import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from streamlit_extras.metric_cards import style_metric_cards
from prophet import Prophet

# Modern dark theme, fonts, and green-accented CSS
st.set_page_config(page_title="Drive & Earn Admin Dashboard", layout="wide", page_icon="🌱")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Montserrat:wght@600&display=swap');
    html, body, [class*='css']  {
        font-family: 'Inter', 'Montserrat', sans-serif;
        background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%);
        color: #e0f2f1;
    }
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #22ff8c !important;
        color: #181f24 !important;
    }
    .stMultiSelect [data-baseweb="tag"] .-remove {
        color: #181f24 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar logo and style
st.sidebar.markdown('<div style="text-align:center;margin-bottom:2rem;"><span style="font-size:2.5rem;">🌱</span><br><span style="font-size:1.3rem;font-weight:700;color:#7fff7f;">Byteme-AI</span></div>', unsafe_allow_html=True)

st.title("Drive & Earn – Admin User Insights Dashboard")
# Add custom CSS to make the main header white and bold
st.markdown("""
<style>
h1, .st-emotion-cache-18ni7ap h1, .st-emotion-cache-10trblm h1 {
    color: #fff !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em;
    text-shadow: 0 2px 8px rgba(0,0,0,0.18);
}
</style>
""", unsafe_allow_html=True)
st.markdown("---")

# --- Custom Plotly Template ---
custom_template = dict(
    layout= dict(
        font=dict(family="Inter, Montserrat, sans-serif", color="#e0f2f1"),
        paper_bgcolor="#000",  # pure black background
        plot_bgcolor="#000",   # pure black background
        title=dict(font=dict(size=22, color="#7fff7f", family="Montserrat, Inter, sans-serif")),
        xaxis=dict(
            gridcolor="rgba(127,255,127,0.10)",
            zerolinecolor="#7fff7f",
            linecolor="#7fff7f",
            tickfont=dict(color="#e0f2f1")
        ),
        yaxis=dict(
            gridcolor="rgba(127,255,127,0.10)",
            zerolinecolor="#7fff7f",
            linecolor="#7fff7f",
            tickfont=dict(color="#e0f2f1")
        ),
        legend=dict(font=dict(color="#e0f2f1")),
        margin=dict(l=40, r=20, t=60, b=40),
        colorway=["#7fff7f", "#4fd1c5", "#63b3ed", "#f6e05e", "#f687b3"]
    )
)
pio.templates["custom_dark_green"] = custom_template

# Remove any Streamlit CSS that could override chart backgrounds
st.markdown("""
<style>
.stPlotlyChart > div > div > svg {
    background: #000 !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------
# Data Loading (from CSV for reproducibility)
# ----------------------
def load_data():
    df = pd.read_csv('mockdata.csv')
    df['photo_timestamp'] = pd.to_datetime(df['photo_timestamp'])
    df['vehicle_number'] = df['vehicle_number'].replace({np.nan: None, '': None})
    return df

def derive_metrics(df):
    df = df.copy()
    df['distance_driven'] = 0
    for user in df['user_id'].unique():
        user_idx = df[df['user_id'] == user].index
        df.loc[user_idx, 'distance_driven'] = df.loc[user_idx, 'odometer_km'].diff().fillna(0)
        df.loc[user_idx, 'distance_driven'] = df.loc[user_idx, 'distance_driven'].clip(lower=0)
    df['reward_tokens'] = (df['distance_driven'] / 10).round(2)
    df['carbon_saved_kg'] = (df['distance_driven'] * 0.12).round(2)
    return df

# Load and process data
df = load_data()
df = derive_metrics(df)

# Sidebar Filters
st.sidebar.header("Filters")
user_filter = st.sidebar.multiselect(
    "Select User(s)", options=df['user_name'].unique(), default=list(df['user_name'].unique())
)
vehicle_type_filter = st.sidebar.multiselect(
    "Select Vehicle Type(s)", options=df['vehicle_type'].unique(), default=list(df['vehicle_type'].unique())
)
date_range = st.sidebar.date_input(
    "Date Range", [df['photo_timestamp'].min().date(), df['photo_timestamp'].max().date()]
)

if isinstance(date_range, list) or isinstance(date_range, tuple):
    if len(date_range) == 1:
        start_date = end_date = date_range[0]
    else:
        start_date, end_date = date_range[0], date_range[1]
else:
    start_date = end_date = date_range

filtered_df = df[
    (df['user_name'].isin(user_filter)) &
    (df['vehicle_type'].isin(vehicle_type_filter)) &
    (df['photo_timestamp'].dt.date >= start_date) &
    (df['photo_timestamp'].dt.date <= end_date)
]

# After filtered_df is created, add this check:
if filtered_df.empty:
    today = pd.Timestamp.today().date()
    if start_date > today:
        st.warning("You have selected a date range in the future. Please select a date range that includes today or earlier.")
    else:
        st.warning("No data available for the selected date range and filters. Please adjust your selection.")
    st.stop()

tab1, tab2 = st.tabs(["Overall Insights", "Per User Insights"])

# --- Modern Metric Cards and Charts in Overall Insights ---
def metric_card(title, value, icon, color):
    return f"<div style='background:rgba(34,255,140,0.10);border-radius:18px;padding:1.2rem 1.5rem 1.1rem 1.5rem;box-shadow:0 2px 16px 0 rgba(0,255,128,0.08);margin-bottom:1.2rem;min-width:180px;'><div style='display:flex;align-items:center;gap:0.7rem;'><span style='font-size:2rem;'>{icon}</span><span style='font-size:1.1rem;color:{color};font-weight:600;'>{title}</span></div><div style='font-size:2.1rem;font-weight:700;margin-top:0.2rem;color:#fff;'>{value}</div></div>"

with tab1:
    # --- Metric Cards ---
    cards_html = (
        '<div style="display:flex;gap:1.5rem;flex-wrap:wrap;">'
        + metric_card("Total Users", filtered_df['user_name'].nunique(), "👥", "#7fff7f")
        + metric_card("Total KM Driven", int(filtered_df['distance_driven'].sum()), "🚗", "#4fd1c5")
        + metric_card("Total Submissions", len(filtered_df), "📸", "#f6e05e")
        + metric_card("Total Tokens Rewarded", round(filtered_df['reward_tokens'].sum(), 2), "🪙", "#f687b3")
        + metric_card("Total CO₂ Saved (kg)", round(filtered_df['carbon_saved_kg'].sum(), 2), "🌱", "#7fff7f")
        + metric_card("Avg CO₂ Saved/User", round(filtered_df.groupby('user_name')['carbon_saved_kg'].sum().mean(), 2), "📊", "#63b3ed")
        + '</div>'
    )
    st.markdown(cards_html, unsafe_allow_html=True)
    st.markdown("---")

    # --- Submission Analytics Charts ---
    st.markdown('<h3 style="color:#fff;font-size:2rem;font-weight:700;display:flex;align-items:center;gap:0.7rem;"><span style="font-size:2rem;">📈</span> Submission Analytics</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        # Submissions Over Time
        sub_per_day = filtered_df.groupby(filtered_df['photo_timestamp'].dt.date).size()
        sub_per_day_df = sub_per_day.reset_index()
        sub_per_day_df.columns = ['Date', 'Submissions']
        fig1 = px.bar(sub_per_day_df, x='Date', y='Submissions', title="Submissions Over Time", template="custom_dark_green")
        fig1.update_layout(
            paper_bgcolor="#000", plot_bgcolor="#000",
            font_color="#fff",
            title_font_color="#fff",
            xaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
            yaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
            legend=dict(font=dict(color="#fff"))
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        # Total Distance per Day
        dist_per_day = filtered_df.groupby(filtered_df['photo_timestamp'].dt.date)['distance_driven'].sum()
        dist_per_day_df = dist_per_day.reset_index()
        dist_per_day_df.columns = ['Date', 'Distance (km)']
        fig2 = px.line(dist_per_day_df, x='Date', y='Distance (km)', title="Total Distance Over Time", template="custom_dark_green", markers=True, line_shape='spline')
        fig2.update_layout(
            paper_bgcolor="#000", plot_bgcolor="#000",
            font_color="#fff",
            title_font_color="#fff",
            xaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
            yaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
            legend=dict(font=dict(color="#fff"))
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # Top 10 Users by KM
        top_users = filtered_df.groupby('user_id')['distance_driven'].sum().sort_values(ascending=False).head(10)
        top_users_df = top_users.reset_index()
        top_users_df['User'] = top_users_df['user_id'].map(dict(zip(df['user_id'], df['user_name'])))
        fig3 = px.bar(top_users_df, x='User', y='distance_driven', title="Top 10 Users by KM Driven", template="custom_dark_green")
        fig3.update_layout(
            paper_bgcolor="#000", plot_bgcolor="#000",
            font_color="#fff",
            title_font_color="#fff",
            xaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
            yaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
            legend=dict(font=dict(color="#fff"))
        )
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        # Frequent Uploaders
        freq_uploaders = filtered_df.groupby('user_id').size().sort_values(ascending=False).head(5)
        freq_uploaders_df = freq_uploaders.reset_index()
        freq_uploaders_df['User'] = freq_uploaders_df['user_id'].map(dict(zip(df['user_id'], df['user_name'])))
        freq_uploaders_df.columns = ['user_id', 'Submissions', 'User']
        fig4 = px.bar(freq_uploaders_df, x='User', y='Submissions', title="Frequent Uploaders", template="custom_dark_green")
        fig4.update_layout(
            paper_bgcolor="#000", plot_bgcolor="#000",
            font_color="#fff",
            title_font_color="#fff",
            xaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
            yaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
            legend=dict(font=dict(color="#fff"))
        )
        st.plotly_chart(fig4, use_container_width=True)

    # --- Carbon & Token Trends ---
    st.markdown('<h3 style="color:#fff;font-size:2rem;font-weight:700;display:flex;align-items:center;gap:0.7rem;"><span style="font-size:2rem;">🌱</span> Carbon & Token Trends</h3>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        # Carbon Saved Trend
        carbon_trend = filtered_df.groupby(filtered_df['photo_timestamp'].dt.date)['carbon_saved_kg'].sum()
        carbon_trend_df = carbon_trend.reset_index()
        carbon_trend_df.columns = ['Date', 'CO₂ Saved (kg)']
        fig5 = px.line(carbon_trend_df, x='Date', y='CO₂ Saved (kg)', title="Carbon Saved Trend", template="custom_dark_green", markers=True, line_shape='spline')
        fig5.update_layout(
            paper_bgcolor="#000", plot_bgcolor="#000",
            font_color="#fff",
            title_font_color="#fff",
            xaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
            yaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
            legend=dict(font=dict(color="#fff"))
        )
        st.plotly_chart(fig5, use_container_width=True)
    with col6:
        # Token Trend
        token_trend = filtered_df.groupby(filtered_df['photo_timestamp'].dt.date)['reward_tokens'].sum()
        token_trend_df = token_trend.reset_index()
        token_trend_df.columns = ['Date', 'Tokens']
        fig6 = px.line(token_trend_df, x='Date', y='Tokens', title="Token Trend", template="custom_dark_green", markers=True, line_shape='spline')
        fig6.update_layout(
            paper_bgcolor="#000", plot_bgcolor="#000",
            font_color="#fff",
            title_font_color="#fff",
            xaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
            yaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
            legend=dict(font=dict(color="#fff"))
        )
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")

    # --- Anomaly Detection Panel ---
    def detect_spikes(df, km_per_day_threshold=200):
        df = df.copy()
        df = df.sort_values(by=['user_id', 'photo_timestamp'])
        df['prev_time'] = df.groupby('user_id')['photo_timestamp'].shift(1)
        df['prev_odo'] = df.groupby('user_id')['odometer_km'].shift(1)
        df['distance_diff'] = df['odometer_km'] - df['prev_odo']
        df['time_diff_days'] = (df['photo_timestamp'] - df['prev_time']).dt.total_seconds() / (60 * 60 * 24)
        df['daily_avg_km'] = df['distance_diff'] / df['time_diff_days']
        df['distance_diff'] = df['distance_diff'].clip(lower=0)
        df['daily_avg_km'] = df['daily_avg_km'].replace([np.inf, -np.inf], np.nan)
        spikes_df = df[df['daily_avg_km'] > km_per_day_threshold]
        return spikes_df[['user_id', 'photo_timestamp', 'distance_diff', 'time_diff_days', 'daily_avg_km']]

    st.markdown('<h3 style="color:#fff;font-size:2rem;font-weight:700;display:flex;align-items:center;gap:0.7rem;"><span style="font-size:2rem;">🚨</span> Anomaly Detection Panel</h3>', unsafe_allow_html=True)
    spike_threshold = 200  # km/day
    spike_table = detect_spikes(filtered_df, km_per_day_threshold=spike_threshold)
    if not spike_table.empty:
        spike_table['User'] = spike_table['user_id'].map(dict(zip(df['user_id'], df['user_name'])))
        spike_table_renamed = spike_table.rename(columns={
            'User': 'User Name',
            'photo_timestamp': 'Photo Upload Date',
            'distance_diff': 'Distance Travelled',
            'time_diff_days': 'Period (Days)',
            'daily_avg_km': 'Daily Avg Km'
        })
        spike_table_renamed.reset_index(drop=True, inplace=True)
        # Render as custom HTML table for black background and white text
        table_html = """
        <style>
        .custom-black-table {
            width: 100%;
            border-collapse: collapse;
            background: #000;
            color: #fff;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 1.5rem;
        }
        .custom-black-table th, .custom-black-table td {
            border: 1px solid #222;
            padding: 0.7em 1.1em;
            text-align: left;
        }
        .custom-black-table th {
            background: #111;
            color: #fff;
            font-weight: 700;
        }
        .custom-black-table tr:nth-child(even) {
            background: #181818;
        }
        .custom-black-table tr:nth-child(odd) {
            background: #000;
        }
        </style>
        <table class="custom-black-table">
            <thead>
                <tr>
                    <th>User Name</th>
                    <th>Photo Upload Date</th>
                    <th>Distance Travelled</th>
                    <th>Period (Days)</th>
                    <th>Daily Avg Km</th>
                </tr>
            </thead>
            <tbody>
        """
        for _, row in spike_table_renamed.iterrows():
            table_html += f"<tr>"
            table_html += f"<td>{row['User Name']}</td>"
            table_html += f"<td>{row['Photo Upload Date']}</td>"
            table_html += f"<td>{int(row['Distance Travelled']) if pd.notnull(row['Distance Travelled']) else ''}</td>"
            table_html += f"<td>{int(row['Period (Days)']) if pd.notnull(row['Period (Days)']) else ''}</td>"
            table_html += f"<td>{round(row['Daily Avg Km'], 1) if pd.notnull(row['Daily Avg Km']) else ''}</td>"
            table_html += "</tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.success("Wow! No anomalies detected!")

    # --- Forecasted Charts ---
    date_range_days = (end_date - start_date).days if 'start_date' in locals() and 'end_date' in locals() else 0
    if date_range_days >= 60:
        st.markdown('<h3 style="color:#fff;font-size:2rem;font-weight:700;display:flex;align-items:center;gap:0.7rem;"><span style="font-size:2rem;">🔮</span> Forecasted Monthly Distance (Next 2 Months)</h3>', unsafe_allow_html=True)
        monthly_distance = (
            filtered_df.groupby(filtered_df['photo_timestamp'].dt.to_period('M'))['distance_driven']
            .sum()
            .reset_index()
        )
        monthly_distance['photo_timestamp'] = monthly_distance['photo_timestamp'].astype(str)
        monthly_distance.columns = ['ds', 'y']
        monthly_distance['ds'] = pd.to_datetime(monthly_distance['ds'])
        from prophet import Prophet
        model = Prophet()
        model.fit(monthly_distance)
        future = model.make_future_dataframe(periods=3, freq='M')
        forecast = model.predict(future)
        forecast_result = forecast[['ds', 'yhat']].copy()
        forecast_result['ds'] = forecast_result['ds'].dt.to_period('M').dt.to_timestamp()
        actual = monthly_distance.copy()
        actual['type'] = 'Actual'
        forecast_result['type'] = ['Actual' if date in actual['ds'].values else 'Forecast' for date in forecast_result['ds']]
        combined_df = pd.concat([
            actual[['ds', 'y', 'type']],
            forecast_result[~forecast_result['ds'].isin(actual['ds'])].rename(columns={'yhat': 'y'})
        ])
        combined_df['Month'] = combined_df['ds'].dt.strftime('%b %Y')
        import plotly.graph_objects as go
        fig = go.Figure()
        for data_type, group in combined_df.groupby('type'):
            fig.add_trace(go.Bar(
                x=group['Month'],
                y=group['y'],
                name=data_type,
                marker_color='#7fff7f' if data_type == 'Actual' else '#63b3ed'
            ))
        fig.update_layout(
            paper_bgcolor="#000", plot_bgcolor="#000",
            font_color="#fff",
            title_font_color="#fff",
            xaxis=dict(title='Month', title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
            yaxis=dict(title='Distance (km)', title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
            barmode='group',
            legend=dict(title="Type", font=dict(color="#fff")),
            margin=dict(l=40, r=20, t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<h3 style="color:#fff;font-size:2rem;font-weight:700;display:flex;align-items:center;gap:0.7rem;"><span style="font-size:2rem;">🔮</span> Forecasted Monthly Uploads (Next 2 Months)</h3>', unsafe_allow_html=True)
        monthly_uploads = filtered_df.groupby(
            pd.Grouper(key='photo_timestamp', freq='M')
        ).size().reset_index(name='uploads')
        monthly_uploads.columns = ['ds', 'y']
        if len(monthly_uploads.dropna()) < 2:
            st.warning("⚠️ Not enough data to generate forecast. Please select a longer date range.")
        else:
            model = Prophet()
            model.fit(monthly_uploads)
            future = model.make_future_dataframe(periods=2, freq='M')
            forecast = model.predict(future)
            forecast_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'Month', 'yhat': 'Uploads'})
            forecast_df['Uploads'] = forecast_df['Uploads'].round().astype(int)
            actual_df = monthly_uploads.rename(columns={'ds': 'Month', 'y': 'Uploads'})
            actual_df['Type'] = 'Actual'
            forecast_df['Type'] = 'Forecast'
            combined_df = pd.concat([
                actual_df,
                forecast_df[forecast_df['Month'] > actual_df['Month'].max()]
            ])
            combined_df['MonthLabel'] = combined_df['Month'].dt.strftime('%b %Y')
            fig2 = go.Figure()
            for t in combined_df['Type'].unique():
                data = combined_df[combined_df['Type'] == t]
                fig2.add_trace(go.Bar(
                    x=data['MonthLabel'],
                    y=data['Uploads'],
                    name=t,
                    marker_color='#63b3ed' if t == 'Forecast' else '#7fff7f',
                ))
            fig2.update_layout(
                paper_bgcolor="#000", plot_bgcolor="#000",
                font_color="#fff",
                title_font_color="#fff",
                xaxis=dict(title='Month', title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
                yaxis=dict(title='Number of Uploads', title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
                barmode='group',
                legend=dict(title="Type", font=dict(color="#fff")),
                margin=dict(l=40, r=20, t=60, b=40)
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Select a date range of at least 60 days to view forecasted distance and uploads charts.")

st.markdown("---")


# ////////

with tab2:
    # Make header white
    st.markdown('<h3 style="color:#fff;font-size:2rem;font-weight:700;display:flex;align-items:center;gap:0.7rem;"><span style="font-size:2rem;">🔍</span> User Drilldown & Insights</h3>', unsafe_allow_html=True)
    # Custom CSS for dark selectbox (input, dropdown, border)
    st.markdown('''<style>
    div[data-baseweb="select"], .stSelectbox, .stSelectbox > div {
        background-color: #181f24 !important;
        color: #fff !important;
        border-radius: 10px !important;
        border: 1.5px solid #333 !important;
    }
    .stSelectbox label, .stSelectbox span {
        color: #fff !important;
    }
    .st-emotion-cache-1c7y2kd, .st-emotion-cache-1c7y2kd label {
        color: #fff !important;
    }
    .st-emotion-cache-1c7y2kd input {
        background: #181f24 !important;
        color: #fff !important;
        border: 1.5px solid #333 !important;
    }
    .stSelectbox [data-baseweb="popover"] {
        background: #181f24 !important;
        color: #fff !important;
        border-radius: 10px !important;
        border: 1.5px solid #333 !important;
    }
    </style>''', unsafe_allow_html=True)
    selected_user_name = st.selectbox("Select a User", options=filtered_df['user_name'].unique())
    user_wallets = df[df['user_name'] == selected_user_name]['wallet_id'].unique()
    if len(user_wallets) == 0:
        st.warning("No accounts found for this user in the selected date range and filters.")
        st.stop()
    if len(user_wallets) > 1:
        selected_wallet = st.selectbox("Select Account (by Wallet)", options=user_wallets)
    else:
        selected_wallet = user_wallets[0]
    selected_user_id = df[(df['user_name'] == selected_user_name) & (df['wallet_id'] == selected_wallet)]['user_id'].iloc[0]
    user_df = filtered_df[(filtered_df['wallet_id'] == selected_wallet) & (filtered_df['user_id'] == selected_user_id)]
    user_info = df[df['wallet_id'] == selected_wallet].iloc[0]

    if user_df.empty:
        st.warning("No data for this user/account in the current filter selection.")
        st.stop()

    # Mask wallet id for privacy
    def mask_wallet(wallet_id):
        if isinstance(wallet_id, str) and len(wallet_id) > 10:
            return wallet_id[:6] + '...' + wallet_id[-4:]
        return wallet_id

    # --- Per User Spike Detection Function (move above usage) ---
    def per_user_spikes(user_df, threshold=200):
        df = user_df.copy().sort_values('photo_timestamp').reset_index(drop=True)
        # If multiple vehicles/wallets per user, group by that too:
        if 'wallet_id' in df.columns:
            df['prev_time'] = df.groupby('wallet_id')['photo_timestamp'].shift(1)
            df['prev_odo'] = df.groupby('wallet_id')['odometer_km'].shift(1)
        else:
            df['prev_time'] = df['photo_timestamp'].shift(1)
            df['prev_odo'] = df['odometer_km'].shift(1)
        df['distance_diff'] = df['odometer_km'] - df['prev_odo']
        df['time_diff_days'] = (df['photo_timestamp'] - df['prev_time']).dt.total_seconds() / (60 * 60 * 24)
        df['daily_avg_km'] = df['distance_diff'] / df['time_diff_days']
        df['distance_diff'] = df['distance_diff'].clip(lower=0)
        df['daily_avg_km'] = df['daily_avg_km'].replace([np.inf, -np.inf], np.nan)
        # Ignore the first row for each group (where prev_odo is NaN)
        spikes = df[(df['daily_avg_km'] > threshold) & df['prev_odo'].notnull()]
        return spikes

    # --- Calculate Per User Summary Metrics (move above layout) ---
    total_distance = int(user_df['distance_driven'].sum())
    avg_km_per_submission = round(user_df['distance_driven'].mean(), 2) if len(user_df) > 0 else 0
    total_tokens = round(user_df['reward_tokens'].sum(), 2)
    total_co2 = round(user_df['carbon_saved_kg'].sum(), 2)
    # --- Responsive Per User Summary Layout: Basic Details + 2x2 Metric Cards (side by side, cards touch, no gap) ---
    st.markdown(f'''
    <div style="display:flex;gap:1.2rem;align-items:stretch;margin-bottom:1.2rem;flex-wrap:wrap;">
        <div style="flex:2.2;min-width:420px;max-width:750px;">
            <div style="background:rgba(34,255,140,0.10);border-radius:16px;padding:1.2rem 1.5rem 1.2rem 1.5rem;box-shadow:0 2px 16px 0 rgba(0,255,128,0.08);margin-bottom:0;width:100%;">
                <div style="font-size:1.2rem;font-weight:700;color:#7fff7f;margin-bottom:0.7rem;">Basic Details</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.7rem 2.5rem;align-items:center;">
                    <div style="color:#7fff7f;font-weight:600;">User Name:</div>
                    <div style="color:#fff;font-weight:500;">{user_info['user_name']}</div>
                    <div style="color:#7fff7f;font-weight:600;">User ID:</div>
                    <div style="color:#fff;font-weight:500;">{user_info['user_id']}</div>
                    <div style="color:#7fff7f;font-weight:600;">Wallet ID:</div>
                    <div style="color:#63b3ed;font-weight:500;">{mask_wallet(user_info['wallet_id'])}</div>
                    <div style="color:#7fff7f;font-weight:600;">Vehicle Type:</div>
                    <div style="color:#fff;font-weight:500;">{user_info['vehicle_type']}</div>
                    <div style="color:#7fff7f;font-weight:600;">Vehicle Number:</div>
                    <div style="color:#fff;font-weight:500;">{user_info['vehicle_number']}</div>
                </div>
            </div>
        </div>
        <div style="flex:2;min-width:520px;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:0.7rem 0.7rem;align-items:stretch;">
            <div style="background:rgba(34,255,140,0.10);border-radius:14px;padding:1.1rem 3.2rem;box-shadow:0 2px 12px 0 rgba(0,255,128,0.08);min-width:240px;min-height:80px;display:flex;flex-direction:column;justify-content:center;">
                <div style="font-size:1.05rem;color:#7fff7f;font-weight:600;">Total Distance Driven</div>
                <div style="font-size:1.7rem;font-weight:700;color:#fff;margin-top:0.2rem;">{total_distance}</div>
            </div>
            <div style="background:rgba(34,255,140,0.10);border-radius:14px;padding:1.1rem 3.2rem;box-shadow:0 2px 12px 0 rgba(0,255,128,0.08);min-width:240px;min-height:80px;display:flex;flex-direction:column;justify-content:center;">
                <div style="font-size:1.05rem;color:#7fff7f;font-weight:600;">Avg KM per Submission</div>
                <div style="font-size:1.7rem;font-weight:700;color:#fff;margin-top:0.2rem;">{avg_km_per_submission}</div>
            </div>
            <div style="background:rgba(34,255,140,0.10);border-radius:14px;padding:1.1rem 3.2rem;box-shadow:0 2px 12px 0 rgba(0,255,128,0.08);min-width:240px;min-height:80px;display:flex;flex-direction:column;justify-content:center;">
                <div style="font-size:1.05rem;color:#7fff7f;font-weight:600;">Total Tokens Earned</div>
                <div style="font-size:1.7rem;font-weight:700;color:#fff;margin-top:0.2rem;">{total_tokens}</div>
            </div>
            <div style="background:rgba(99,179,237,0.10);border-radius:14px;padding:1.1rem 3.2rem;box-shadow:0 2px 12px 0 rgba(99,179,237,0.08);min-width:240px;min-height:80px;display:flex;flex-direction:column;justify-content:center;">
                <div style="font-size:1.05rem;color:#63b3ed;font-weight:600;">Total CO<sub>2</sub> Saved</div>
                <div style="font-size:1.7rem;font-weight:700;color:#fff;margin-top:0.2rem;">{total_co2} kg</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # --- Odometer Spike/Anomaly Card (full width, aligned with above row) ---
    user_spikes = per_user_spikes(user_df)
    if user_spikes.empty:
        st.markdown("""
        <div style='display:flex;width:100%;max-width:none;'>
            <div style='background:rgba(34,255,140,0.10);border-radius:14px;padding:1.1rem 1.3rem 1.1rem 1.3rem;box-shadow:0 2px 12px 0 rgba(0,255,128,0.08);margin-bottom:1.1rem;width:100%;'>
                <span style='color:#7fff7f;font-size:1.25rem;font-weight:700;'>No odometer spikes detected for this user.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='display:flex;width:100%;max-width:none;'>
            <div style='background:rgba(255,80,80,0.13);border-radius:14px;padding:1.1rem 1.3rem 1.1rem 1.3rem;box-shadow:0 2px 12px 0 rgba(255,80,80,0.08);margin-bottom:1.1rem;width:100%;'>
                <span style='color:#ff5050;font-size:1.25rem;font-weight:700;'>Odometer spike(s) detected! Please review the user's submissions for anomalies.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Show spike table for this user
        spike_table = user_spikes.copy()
        spike_table_renamed = spike_table.rename(columns={
            'photo_timestamp': 'Photo Upload Date',
            'distance_diff': 'Distance Travelled',
            'time_diff_days': 'Period (Days)',
            'daily_avg_km': 'Daily Avg Km'
        })
        spike_table_renamed.reset_index(drop=True, inplace=True)
        table_html = """
        <style>
        .custom-black-table-user {
            width: 100%;
            border-collapse: collapse;
            background: #000;
            color: #fff;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 1.5rem;
        }
        .custom-black-table-user th, .custom-black-table-user td {
            border: 1px solid #222;
            padding: 0.7em 1.1em;
            text-align: left;
        }
        .custom-black-table-user th {
            background: #111;
            color: #fff;
            font-weight: 700;
        }
        .custom-black-table-user tr:nth-child(even) {
            background: #181818;
        }
        .custom-black-table-user tr:nth-child(odd) {
            background: #000;
        }
        </style>
        <table class="custom-black-table-user">
            <thead>
                <tr>
                    <th>Photo Upload Date</th>
                    <th>Distance Travelled</th>
                    <th>Period (Days)</th>
                    <th>Daily Avg Km</th>
                </tr>
            </thead>
            <tbody>
        """
        for _, row in spike_table_renamed.iterrows():
            table_html += f"<tr>"
            table_html += f"<td>{row['Photo Upload Date']}</td>"
            table_html += f"<td>{int(row['Distance Travelled']) if pd.notnull(row['Distance Travelled']) else ''}</td>"
            table_html += f"<td>{int(row['Period (Days)']) if pd.notnull(row['Period (Days)']) else ''}</td>"
            table_html += f"<td>{round(row['Daily Avg Km'], 1) if pd.notnull(row['Daily Avg Km']) else ''}</td>"
            table_html += "</tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

    # --- Tier Logic ---
    total_km = user_df['distance_driven'].sum()
    total_uploads = len(user_df)
    # Tier thresholds
    tier1_km, tier1_uploads = 100, 5
    tier2_km, tier2_uploads = 1000, 20
    tier3_km, tier3_uploads = 5000, 40
    if total_km >= tier3_km and total_uploads >= tier3_uploads:
        tier = "Tier 3"
        next_tier_msg = "Max tier achieved!"
        tier_progress = 100
    elif total_km >= tier2_km and total_uploads >= tier2_uploads:
        tier = "Tier 2"
        km_left = max(0, tier3_km - int(total_km))
        uploads_left = max(0, tier3_uploads - total_uploads)
        tier_progress = min(100, int(100 * total_km / tier3_km), int(100 * total_uploads / tier3_uploads))
        if km_left > 0 and uploads_left > 0:
            next_tier_msg = f"{km_left} km and {uploads_left} uploads to reach Tier 3"
        elif km_left > 0:
            next_tier_msg = f"{km_left} km to reach Tier 3 (Uploads criteria met!)"
        elif uploads_left > 0:
            next_tier_msg = f"{uploads_left} uploads to reach Tier 3 (Distance criteria met!)"
        else:
            next_tier_msg = "Max tier achieved!"
    elif total_km >= tier1_km and total_uploads >= tier1_uploads:
        tier = "Tier 1"
        km_left = max(0, tier2_km - int(total_km))
        uploads_left = max(0, tier2_uploads - total_uploads)
        tier_progress = min(100, int(100 * total_km / tier2_km), int(100 * total_uploads / tier2_uploads))
        if km_left > 0 and uploads_left > 0:
            next_tier_msg = f"{km_left} km and {uploads_left} uploads to reach Tier 2"
        elif km_left > 0:
            next_tier_msg = f"{km_left} km to reach Tier 2 (Uploads criteria met!)"
        elif uploads_left > 0:
            next_tier_msg = f"{uploads_left} uploads to reach Tier 2 (Distance criteria met!)"
        else:
            next_tier_msg = "Max tier achieved!"
    else:
        tier = "Unranked"
        km_left = max(0, tier1_km - int(total_km))
        uploads_left = max(0, tier1_uploads - total_uploads)
        tier_progress = min(100, int(100 * total_km / tier1_km), int(100 * total_uploads / tier1_uploads))
        if km_left > 0 and uploads_left > 0:
            next_tier_msg = f"{km_left} km and {uploads_left} uploads to reach Tier 1"
        elif km_left > 0:
            next_tier_msg = f"{km_left} km to reach Tier 1 (Uploads criteria met!)"
        elif uploads_left > 0:
            next_tier_msg = f"{uploads_left} uploads to reach Tier 1 (Distance criteria met!)"
        else:
            next_tier_msg = "Almost there!"
    # Tier Card with Progress Bar
    st.markdown(f"""
    <div style='background:rgba(34,255,140,0.10);border-radius:14px;padding:1.1rem 1.3rem 1.1rem 1.3rem;box-shadow:0 2px 12px 0 rgba(0,255,128,0.08);margin-bottom:1.1rem;min-width:180px;'>
        <div style='display:flex;align-items:center;gap:0.7rem;'>
            <span style='font-size:1.7rem;'>🏆</span>
            <span style='font-size:1.1rem;color:#7fff7f;font-weight:600;'>Current Tier</span>
        </div>
        <div style='font-size:1.7rem;font-weight:700;margin-top:0.2rem;color:#fff;'>{tier}</div>
        <div style='background:#222;border-radius:8px;height:8px;width:100%;margin-top:10px;margin-bottom:6px;'>
            <div style='background:#7fff7f;height:8px;border-radius:8px;width:{tier_progress}%;transition:width 0.6s;'></div>
        </div>
        <div style='font-size:1rem;color:#e0f2f1;margin-top:2px;'>{next_tier_msg}</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Badge Logic ---
    badges = []
    badge_progress = []
    # Define badge criteria
    badge_defs = [
        {"name": "Rookie Rider", "desc": "Travelled 100km", "icon": "🚦", "achieved": total_km >= 100, "progress": min(100, int(100*total_km/100))},
        {"name": "Road Runner", "desc": "Travelled 1000km", "icon": "🏁", "achieved": total_km >= 1000, "progress": min(100, int(100*total_km/1000))},
        {"name": "Long Haul", "desc": "Travelled 5000km", "icon": "🛣️", "achieved": total_km >= 5000, "progress": min(100, int(100*total_km/5000))},
        {"name": "Consistent Contributor", "desc": "20 uploads", "icon": "📅", "achieved": total_uploads >= 20, "progress": min(100, int(100*total_uploads/20))},
        {"name": "Active Days", "desc": "Uploaded on 10 different days", "icon": "📆", "achieved": user_df['photo_timestamp'].dt.date.nunique() >= 10, "progress": min(100, int(100*user_df['photo_timestamp'].dt.date.nunique()/10))},
        {"name": "Savings Hero", "desc": "Saved 20kg CO₂", "icon": "🌱", "achieved": user_df['carbon_saved_kg'].sum() >= 20, "progress": min(100, int(100*user_df['carbon_saved_kg'].sum()/20))},
        {"name": "Battery Saver", "desc": "Avg battery > 80%", "icon": "🔋", "achieved": user_df['battery_percentage'].mean() >= 80, "progress": min(100, int(100*user_df['battery_percentage'].mean()/80))},
    ]
    badge_html = "<div style='display:flex;gap:1.2rem;flex-wrap:wrap;margin-bottom:1.2rem;'>"
    for b in badge_defs:
        check = "<span style='color:#7fff7f;font-size:1.3rem;margin-left:6px;'>✔️</span>" if b["achieved"] else ""
        badge_html += f"<div style='background:rgba(34,255,140,0.10);border-radius:14px;padding:0.9rem 1.1rem 0.9rem 1.1rem;box-shadow:0 2px 12px 0 rgba(0,255,128,0.08);min-width:140px;max-width:180px;'>"
        badge_html += f"<div style='display:flex;align-items:center;gap:0.5rem;'><span style='font-size:1.5rem;'>{b['icon']}</span><span style='font-size:1.05rem;color:#7fff7f;font-weight:600;'>{b['name']}{check}</span></div>"
        badge_html += f"<div style='font-size:0.98rem;color:#e0f2f1;margin-top:2px;'>{b['desc']}</div>"
        badge_html += f"<div style='background:#222;border-radius:8px;height:7px;width:100%;margin-top:8px;'>"
        badge_html += f"<div style='background:{'#7fff7f' if b['achieved'] else '#63b3ed'};height:7px;border-radius:8px;width:{b['progress']}%;transition:width 0.6s;'></div>"
        badge_html += "</div>"
        badge_html += "</div>"
    badge_html += "</div>"
    st.markdown("<div style='font-size:1.1rem;color:#fff;font-weight:600;margin-bottom:0.5rem;'>Badges Earned</div>" + badge_html, unsafe_allow_html=True)

    # --- User Activity & Trends Header ---
    st.markdown(
        '<h3 style="color:#fff;font-size:2rem;font-weight:700;display:flex;align-items:center;gap:0.7rem;">'
        '<span style="font-size:2rem;">📊</span> User Activity & Trends</h3>',
        unsafe_allow_html=True
    )
    # Now show the charts (trendlines green)
    fig5 = px.line(user_df, x='photo_timestamp', y='distance_driven', title="Submission Timeline (KM per Submission)", markers=True, template="custom_dark_green", line_shape='spline')
    fig5.update_traces(line_color="#7fff7f")
    fig5.update_layout(
        paper_bgcolor="#000", plot_bgcolor="#000",
        font_color="#fff",
        title_font_color="#fff",
        xaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
        yaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
        legend=dict(font=dict(color="#fff"))
    )
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.line(user_df, x='photo_timestamp', y='reward_tokens', title="Token Earned Over Time", markers=True, template="custom_dark_green", line_shape='spline')
    fig6.update_traces(line_color="#7fff7f")
    fig6.update_layout(
        paper_bgcolor="#000", plot_bgcolor="#000",
        font_color="#fff",
        title_font_color="#fff",
        xaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
        yaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
        legend=dict(font=dict(color="#fff"))
    )
    st.plotly_chart(fig6, use_container_width=True)

    fig7 = px.line(user_df, x='photo_timestamp', y='battery_percentage', title="Battery % Trend", markers=True, template="custom_dark_green", line_shape='spline')
    fig7.update_traces(line_color="#7fff7f")
    fig7.update_layout(
        paper_bgcolor="#000", plot_bgcolor="#000",
        font_color="#fff",
        title_font_color="#fff",
        xaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
        yaxis=dict(title_font=dict(color="#fff"), tickfont=dict(color="#fff")),
        legend=dict(font=dict(color="#fff"))
    )
    st.plotly_chart(fig7, use_container_width=True)

st.markdown("---")
st.caption("Built for the Drive & Earn Hackathon | Powered by Streamlit, Plotly, and streamlit-extras") 