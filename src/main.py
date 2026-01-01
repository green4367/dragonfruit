from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import os
import sys
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'dragonfruit_secret_key'
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def detect_csv_structure(columns):
    plan_cols = [
        "DateTime", "Style", "Repetitions", "Sets", "Weight"
    ]
    combined_cols = [
        "DateTime", "Training", "Duration", "Active Calories", "Total Calories", "Average Hertrage"
    ]
    if columns == plan_cols:
        return "PlanVersion"
    elif columns == combined_cols:
        return "CombinedVersion"
    else:
        return "unknown"

def calculate_planversion_score(df):
    df = df.copy()
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
    df = df.sort_values('DateTime')
    weight_increments = df['Weight'].diff().fillna(0) > 0
    increment_indices = [0] + list(df.index[weight_increments])
    sessions_per_increment = []
    for i in range(1, len(increment_indices)):
        prev_idx = increment_indices[i-1]
        curr_idx = increment_indices[i]
        sessions = df.index.get_loc(curr_idx) - df.index.get_loc(prev_idx)
        sessions_per_increment.append(sessions)
    if sessions_per_increment:
        avg_sessions = sum(sessions_per_increment) / len(sessions_per_increment)
    else:
        avg_sessions = len(df)
    if avg_sessions <= 4:
        prog_score = 100
    elif avg_sessions <= 7:
        prog_score = int(100 - 10 * (avg_sessions - 4))
    else:
        prog_score = max(0, int(70 - 20 * (avg_sessions - 7)))
    total_days = (df['DateTime'].max() - df['DateTime'].min()).days + 1
    weeks = total_days / 7 if total_days > 0 else 1
    sessions_per_week = len(df) / weeks if weeks > 0 else 0
    if sessions_per_week >= 3:
        consistency_bonus = 10
    elif sessions_per_week >= 2:
        consistency_bonus = 5
    elif sessions_per_week > 1:
        consistency_bonus = 0
    else:
        consistency_bonus = -10
    score = prog_score + consistency_bonus
    score = min(100, max(0, score))
    return score

def parse_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if len(lines) < 4:
        raise ValueError("CSV file is too short to match any known structure.")
    title = lines[0].strip()
    subtitle = lines[1].strip()
    raw_columns = [c.strip() for c in lines[2].strip().split(';')]
    columns = [c for c in raw_columns if c]
    data_lines = lines[3:]
    cleaned_data_lines = [l.rstrip().rstrip(';') + '\n' for l in data_lines if l.strip()]
    from io import StringIO
    data_str = ''.join(cleaned_data_lines)
    df = pd.read_csv(StringIO(data_str), sep=';', names=columns)
    return title, subtitle, columns, df

def plot_planversion(title, subtitle, df, score):
    import plotly.express as px
    import numpy as np
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    # Weight progression
    fig = go.Figure()
    unique_styles = df['Style'].unique()
    style_colors = px.colors.qualitative.Plotly
    color_map = {style: style_colors[i % len(style_colors)] for i, style in enumerate(unique_styles)}
    for style in unique_styles:
        mask = df['Style'] == style
        fig.add_trace(go.Scatter(
            x=df.loc[mask, 'DateTime'],
            y=df.loc[mask, 'Weight'],
            mode='markers+lines',
            name=f"{style} Weight",
            marker=dict(color=color_map[style]),
            line=dict(color=color_map[style])
        ))
    fig.update_layout(
        title=f"{title}<br><sub>{subtitle}</sub>",
        xaxis_title="Date",
        yaxis_title="Weight (kg)",
        template="plotly_dark",
        legend_title="Style"
    )
    # Repetitions progression
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df['DateTime'],
        y=df['Repetitions'],
        mode='lines+markers',
        name='Repetitions',
        marker=dict(color='#e67e22')
    ))
    fig2.update_layout(
        title="Repetitions Progression",
        xaxis_title="Date",
        yaxis_title="Repetitions",
        template="plotly_dark"
    )
    # Score circle as a gauge
    gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "Progression Score"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "#27ae60" if score > 66 else ("#f1c40f" if score > 33 else "#e74c3c")}}
    ))
    gauge.update_layout(template="plotly_dark", height=300)
    return pio.to_html(gauge, full_html=False), pio.to_html(fig, full_html=False), pio.to_html(fig2, full_html=False)

def plot_combinedversion(title, subtitle, df):
    import plotly.express as px
    import numpy as np
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    # Stacked bar chart: Calories by Training Type Over Time
    pivot = df.pivot_table(index='DateTime', columns='Training', values='Total Calories', aggfunc='sum', fill_value=0)
    pivot = pivot.sort_index()
    fig = go.Figure()
    for col in pivot.columns:
        fig.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[col],
            name=col
        ))
    fig.update_layout(
        barmode='stack',
        title=f"{title}<br><sub>{subtitle}</sub><br>Calories Burned by Activity",
        xaxis_title="Date",
        yaxis_title="Calories Burned",
        template="plotly_dark"
    )
    # Heatmap: Activity Frequency (Sessions per Day of Week)
    df['Week'] = df['DateTime'].dt.isocalendar().week
    df['Day'] = df['DateTime'].dt.dayofweek
    heatmap_data = df.groupby(['Week', 'Day']).size().unstack(fill_value=0)
    z = heatmap_data.values.T
    fig2 = go.Figure(data=go.Heatmap(
        z=z,
        x=heatmap_data.index,
        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        colorscale='YlOrRd',
        colorbar=dict(title='Sessions')
    ))
    fig2.update_layout(
        title="Activity Frequency Heatmap",
        xaxis_title="Week",
        yaxis_title="Day of Week",
        template="plotly_dark"
    )
    return pio.to_html(fig, full_html=False), pio.to_html(fig2, full_html=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            try:
                title, subtitle, columns, df = parse_csv(file_path)
                structure = detect_csv_structure(columns)
                if structure == "PlanVersion":
                    # Ensure DateTime is datetime
                    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
                    score = calculate_planversion_score(df)
                    gauge_html, graph1_html, graph2_html = plot_planversion(title, subtitle, df, score)
                    return render_template('index.html', gauge_html=gauge_html, graph1_html=graph1_html, graph2_html=graph2_html)
                elif structure == "CombinedVersion":
                    graph1_html, graph2_html = plot_combinedversion(title, subtitle, df)
                    return render_template('index.html', gauge_html=None, graph1_html=graph1_html, graph2_html=graph2_html)
                else:
                    flash('Unknown CSV structure.')
            except Exception as e:
                flash(f'Error processing file: {e}')
    return render_template('index.html', gauge_html=None, graph1_html=None, graph2_html=None)

if __name__ == "__main__":
    app.run(debug=True)