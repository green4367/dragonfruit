# Entry point for the CSV Analyzer UI application

import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QHBoxLayout, QFrame, QMessageBox
)
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt
import qtmodern.styles
import qtmodern.windows
# Matplotlib for embedding plots
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ScoreCircle(QWidget):
    def __init__(self, score, parent=None):
        super().__init__(parent)
        self.score = score
        self.setMinimumSize(140, 140)

    def setScore(self, score):
        self.score = score
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(10, 10, -10, -10)
        # Color logic
        if self.score <= 33:
            color = QColor("#e74c3c")
        elif self.score <= 66:
            color = QColor("#f1c40f")
        else:
            color = QColor("#27ae60")
        # Draw background circle
        painter.setPen(QPen(QColor("#444"), 12))
        painter.drawEllipse(rect)
        # Draw arc for score
        painter.setPen(QPen(color, 12))
        span_angle = int(360 * 16 * self.score / 100)
        painter.drawArc(rect, 90 * 16, -span_angle)
        # Draw score text
        painter.setPen(QPen(color, 1))
        painter.setFont(QFont("Segoe UI", 32, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, str(self.score))
        # Draw label inside the circle, smaller and near the bottom
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor("#bbb"), 1))
        label_rect = rect.adjusted(0, int(rect.height() * 0.45), 0, -20)
        painter.drawText(label_rect, Qt.AlignHCenter | Qt.AlignBottom, "Progression")
        label_rect = rect.adjusted(0, int(rect.height() * 0.45), 0, -10)
        painter.drawText(label_rect, Qt.AlignHCenter | Qt.AlignBottom, "Score")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trainings Analyzer - Dragonfruit")
        base_width, base_height = 600, 400
        self.resize(int(base_width * 1.3), int(base_height * 1.3))
        self.setMinimumSize(600, 400)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Stylish title
        self.title_label = QLabel("Dragonfruit Trainings Analyzer")
        self.title_label.setFont(QFont("Segoe UI", 22, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)

        # Subtle divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(divider)

        # Info label
        self.info_label = QLabel("Select a CSV file to begin.")
        self.info_label.setFont(QFont("Segoe UI", 12))
        self.info_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.info_label)

        # File picker row
        file_row = QHBoxLayout()
        self.file_button = QPushButton("Choose CSV File")
        self.file_button.setFont(QFont("Segoe UI", 11))
        self.file_button.setIcon(QIcon.fromTheme("document-open"))
        self.file_button.setStyleSheet("padding: 8px 24px; border-radius: 8px; background: #0078d7; color: white;")
        self.file_button.clicked.connect(self.open_file_dialog)
        file_row.addStretch()
        file_row.addWidget(self.file_button)
        file_row.addStretch()
        self.layout.addLayout(file_row)

        # Central content container for score and graph, vertically centered
        self.score_circle = None
        self.score_container = QWidget()
        score_layout = QVBoxLayout()
        score_layout.setContentsMargins(0, 16, 0, 0)
        score_layout.setAlignment(Qt.AlignCenter)
        self.score_container.setLayout(score_layout)

        self.central_content = QWidget()
        self.central_content_layout = QVBoxLayout()
        self.central_content_layout.setAlignment(Qt.AlignCenter)
        self.central_content.setLayout(self.central_content_layout)
        self.central_content_layout.addWidget(self.score_container)
        self.layout.addStretch()
        self.layout.addWidget(self.central_content)
        self.layout.addStretch()

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if file_name:
            self.load_and_detect_csv(file_name)
        else:
            self.info_label.setText("No file selected.")

    def load_and_detect_csv(self, file_path):
        try:
            # Read all lines to handle custom header
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) < 4:
                raise ValueError("CSV file is too short to match any known structure.")
            title = lines[0].strip()
            subtitle = lines[1].strip()
            # Remove empty column names and trailing semicolons
            raw_columns = [c.strip() for c in lines[2].strip().split(';')]
            columns = [c for c in raw_columns if c]
            data_lines = lines[3:]
            # Remove trailing semicolons from data lines
            cleaned_data_lines = [l.rstrip().rstrip(';') + '\n' for l in data_lines if l.strip()]
            from io import StringIO
            data_str = ''.join(cleaned_data_lines)
            df = pd.read_csv(StringIO(data_str), sep=';', names=columns)
        except Exception as e:
            self.show_error(f"Failed to load CSV file.\nError: {str(e)}")
            self.info_label.setText("Error loading file.")
            return

        # Detect structure
        structure = self.detect_csv_structure(columns)
        if structure == "unknown":
            self.show_error("The selected CSV file does not match any known structure.")
            self.info_label.setText("Unknown CSV structure.")
        else:
            self.info_label.setText(f"Loaded {structure} CSV file.")
            self.display_graphs(structure, title, subtitle, df)

    def clear_graphs(self):
        # Remove previous graphs if any
        if hasattr(self, 'canvas') and self.canvas:
            if self.graph_container:
                self.central_content_layout.removeWidget(self.graph_container)
                self.graph_container.setParent(None)
            self.canvas = None
        # Remove score circle widget if present
        if self.score_circle:
            self.score_container.layout().removeWidget(self.score_circle)
            self.score_circle.setParent(None)
            self.score_circle = None

    def display_graphs(self, structure, title, subtitle, df):
        import matplotlib.dates as mdates
        self.clear_graphs()
        def set_spines_white(ax):
            for spine in ax.spines.values():
                spine.set_color('white')

        if structure == "PlanVersion":
            try:
                date_format = "%Y-%m-%d"
                display_format = "%Y-%m-%d"
                df['DateTime'] = pd.to_datetime(df['DateTime'], format=date_format)
                # --- Calculate progression score ---
                score = self.calculate_planversion_score(df)
                # --- Show score in a prominent circle above the graphs ---
                self.score_circle = ScoreCircle(score)
                self.score_container.layout().addWidget(self.score_circle, alignment=Qt.AlignCenter)
                # --- Graphs ---
                fig = Figure(figsize=(7, 6))
                axs = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
                # Top: Weight progression, colored by STYLE
                unique_styles = df['Style'].unique()
                style_colors = {style: f'C{i%10}' for i, style in enumerate(unique_styles)}
                # Handle '-' in Weight: treat as previous value, mark for highlight
                weight_raw = df['Weight'].astype(str)
                dash_mask = weight_raw == '-'
                # Replace '-' with NaN, then forward fill
                df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
                df['Weight'] = df['Weight'].ffill()
                # Plot lines connecting all points (in default color)
                axs[0].plot(df['DateTime'], df['Weight'], color='#555', linewidth=1, alpha=0.5, zorder=1)
                # Plot each node with color by style
                for style in unique_styles:
                    mask = df['Style'] == style
                    axs[0].scatter(df.loc[mask & ~dash_mask, 'DateTime'], df.loc[mask & ~dash_mask, 'Weight'],
                                   color=style_colors[style], label=style, s=60, zorder=2, edgecolor='white', linewidth=0.7)
                # Highlight points where weight was a dash (bodyweight session)
                if dash_mask.any():
                    axs[0].scatter(df.loc[dash_mask, 'DateTime'], df.loc[dash_mask, 'Weight'],
                                   color='none', edgecolor='red', s=80, linewidth=2, marker='o', label='Bodyweight (no weight)')
                axs[0].set_ylabel('Weight (kg)', color='white')
                axs[0].set_title(f"{title}\n{subtitle}", color='white')
                axs[0].grid(True, linestyle='--', alpha=0.5)
                # Ensure y-axis fits all data points if valid
                if not df['Weight'].isnull().all():
                    min_weight = df['Weight'].min()
                    max_weight = df['Weight'].max()
                    # If the range is too small, force to min/max
                    if max_weight - min_weight < 5:
                        axs[0].set_ylim(min_weight - 1, max_weight + 1)
                    else:
                        axs[0].set_ylim(min_weight - (max_weight-min_weight)*0.05, max_weight + (max_weight-min_weight)*0.05)
                # Set tick labels to white
                axs[0].tick_params(axis='x', colors='white')
                axs[0].tick_params(axis='y', colors='white')
                # Custom legend for styles
                handles, labels = axs[0].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                legend = axs[0].legend(by_label.values(), by_label.keys(), loc='upper left', title='Style', fontsize=9, title_fontsize=10)
                if legend:
                    legend.get_title().set_color('white')
                    for text in legend.get_texts():
                        text.set_color('white')
                set_spines_white(axs[0])
                # Bottom: Repetitions progression
                axs[1].plot(df['DateTime'], df['Repetitions'], marker='s', color='#e67e22', label='Repetitions')
                axs[1].set_ylabel('Repetitions', color='white')
                axs[1].set_xlabel('Date', color='white')
                axs[1].legend(loc='upper left')
                axs[1].grid(True, linestyle='--', alpha=0.5)
                for i in range(len(df)):
                    if df['Repetitions'].iloc[i] == 12:
                        axs[1].scatter(df['DateTime'].iloc[i], df['Repetitions'].iloc[i], color='red', s=60, zorder=5, label='12 Reps' if i==0 else "")
                axs[1].xaxis.set_major_formatter(mdates.DateFormatter(display_format))
                fig.autofmt_xdate()
                # Set tick labels to white
                axs[1].tick_params(axis='x', colors='white')
                axs[1].tick_params(axis='y', colors='white')
                # Set legend text color to white
                leg1 = axs[1].get_legend()
                if leg1:
                    for text in leg1.get_texts():
                        text.set_color('white')
                set_spines_white(axs[1])
                # Remove duplicate legend entries
                handles, labels = axs[0].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                axs[0].legend(by_label.values(), by_label.keys(), loc='upper left')
                handles, labels = axs[1].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                axs[1].legend(by_label.values(), by_label.keys(), loc='upper left')
            except Exception as e:
                fig = Figure(figsize=(7, 6))
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Error plotting: {e}", ha='center', va='center')
        elif structure == "CombinedVersion":
            try:
                import numpy as np
                import matplotlib.dates as mdates
                import calendar
                date_format = "%Y-%m-%d"
                display_format = "%Y-%m-%d"
                df['DateTime'] = pd.to_datetime(df['DateTime'], format=date_format)
                # --- Stacked Bar Chart: Calories by Training Type Over Time ---
                pivot = df.pivot_table(index='DateTime', columns='Training', values='Total Calories', aggfunc='sum', fill_value=0)
                pivot = pivot.sort_index()
                fig = Figure(figsize=(10, 7))
                axs = fig.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
                # Stacked bar chart
                bottom = np.zeros(len(pivot))
                colors = [f'C{i}' for i in range(len(pivot.columns))]
                for idx, (col, color) in enumerate(zip(pivot.columns, colors)):
                    axs[0].bar(pivot.index, pivot[col], bottom=bottom, label=col, color=color)
                    bottom += pivot[col].values
                axs[0].set_ylabel('Calories Burned', color='white')
                axs[0].set_title(f"{title}\n{subtitle}\nCalories Burned by Activity", color='white')
                axs[0].legend(loc='upper left', fontsize=8)
                axs[0].grid(True, linestyle='--', alpha=0.5)
                axs[0].tick_params(axis='x', colors='white')
                axs[0].tick_params(axis='y', colors='white')
                leg0 = axs[0].get_legend()
                if leg0:
                    for text in leg0.get_texts():
                        text.set_color('white')
                set_spines_white(axs[0])
                axs[0].xaxis.set_major_formatter(mdates.DateFormatter(display_format))
                fig.autofmt_xdate()
                # --- Heatmap: Activity Frequency (Sessions per Day of Week) ---
                df['Week'] = df['DateTime'].dt.isocalendar().week
                df['Day'] = df['DateTime'].dt.dayofweek
                heatmap_data = df.groupby(['Week', 'Day']).size().unstack(fill_value=0)
                # Ensure all days are present
                for d in range(7):
                    if d not in heatmap_data.columns:
                        heatmap_data[d] = 0
                heatmap_data = heatmap_data.sort_index(axis=1)
                im = axs[1].imshow(heatmap_data.T, aspect='auto', cmap='YlOrRd', origin='lower')
                axs[1].set_yticks(range(7))
                axs[1].set_yticklabels([calendar.day_abbr[d] for d in range(7)], color='white')
                axs[1].set_xticks(range(len(heatmap_data.index)))
                axs[1].set_xticklabels(heatmap_data.index, color='white', rotation=45)
                axs[1].set_xlabel('Week', color='white')
                axs[1].set_ylabel('Day', color='white')
                axs[1].set_title('Activity Frequency Heatmap', color='white')
                set_spines_white(axs[1])
                axs[1].tick_params(axis='x', colors='white')
                axs[1].tick_params(axis='y', colors='white')
                # Add colorbar
                cbar = fig.colorbar(im, ax=axs[1], orientation='vertical', pad=0.02)
                cbar.ax.yaxis.set_tick_params(color='white')
                plt = fig.canvas.figure
                for l in cbar.ax.yaxis.get_ticklabels():
                    l.set_color('white')
            except Exception as e:
                fig = Figure(figsize=(10, 7))
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Error plotting: {e}", ha='center', va='center', color='white')
        else:
            fig = Figure(figsize=(7, 4))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No graph for this structure", ha='center', va='center', color='white')
        # Set transparent background for the figure, axes, and canvas
        fig.patch.set_facecolor('none')
        fig.patch.set_alpha(0.0)
        for ax in fig.get_axes():
            ax.set_facecolor('none')
            ax.patch.set_alpha(0.0)
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background: transparent;")
        # Add margin around the graph
        self.graph_container = QWidget()
        graph_layout = QVBoxLayout()
        graph_layout.setContentsMargins(32, 0, 32, 24)  # left, top, right, bottom
        graph_layout.addWidget(self.canvas)
        self.graph_container.setLayout(graph_layout)
        self.central_content_layout.addWidget(self.graph_container)

    def calculate_planversion_score(self, df):
        # New progression score logic based on weight increments and consistency
        df = df.copy()
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
        df = df.sort_values('DateTime')
        # Find indices where weight increases
        weight_increments = df['Weight'].diff().fillna(0) > 0
        increment_indices = [0] + list(df.index[weight_increments])
        # Calculate sessions per increment
        sessions_per_increment = []
        for i in range(1, len(increment_indices)):
            prev_idx = increment_indices[i-1]
            curr_idx = increment_indices[i]
            sessions = df.index.get_loc(curr_idx) - df.index.get_loc(prev_idx)
            sessions_per_increment.append(sessions)
        # If no increments, use total sessions as denominator
        if sessions_per_increment:
            avg_sessions = sum(sessions_per_increment) / len(sessions_per_increment)
        else:
            avg_sessions = len(df)
        # Map avg_sessions to score
        if avg_sessions <= 4:
            prog_score = 100
        elif avg_sessions <= 7:
            prog_score = int(100 - 10 * (avg_sessions - 4))
        else:
            prog_score = max(0, int(70 - 20 * (avg_sessions - 7)))
        # Consistency bonus/penalty
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

    def detect_csv_structure(self, columns):
        # PlanVersion structure
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

    def show_error(self, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText(message)
        msg.exec_()

def main():
    app = QApplication(sys.argv)
    qtmodern.styles.dark(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
