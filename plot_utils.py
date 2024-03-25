###################################################
# THIS SCRIPT IS TO CONTAIN VISUALIZATION FUNCTIONS
###################################################

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import zscore


def plot_OutReachOutcome_by_Outreach_Attempt(df_outreach_0: pd.DataFrame,
                                             df_outreach_1: pd.DataFrame):
    trace1 = go.Histogram(x=df_outreach_1['OutReachOutcome'],
                          name='Outreach Attempted (Yes)')
    trace2 = go.Histogram(x=df_outreach_0['OutReachOutcome'],
                          name='Outreach Attempted (No)')

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)

    fig.update_layout(
        title={
            'text': 'OutReachOutcome Distribution by Outreach Attempt',
            'y':
            0.95,  # Adjusts the title's position on the y-axis to center vertically
            'x': 0.5,  # Centers the title horizontally
            'xanchor': 'center',
            'yanchor': 'top'
        },
        font=dict(size=18))

    # Update x-axis titles for each subplot individually
    fig.update_xaxes(title_text='', row=1, col=1)  # Update for the first subplot
    fig.update_xaxes(title_text='', row=1, col=2)  # Update for the second subplot

    fig.show()


def plot_categorical_feature(plot_df: pd.DataFrame, cat_feature: str):
    fig = px.histogram(plot_df,
                       x=cat_feature,
                       title=f'Histogram for {cat_feature}')
    # Updating layout for customizations
    fig.update_layout(
        title={
            'text': f'Histogram for {cat_feature}',
            'x': 0.5,
            'xanchor': 'center'
        },  # Centering title
        font=dict(size=18),  # Adjusting font size globally
        width=1200,
        bargap=0.2)
    fig.show()


def plot_dist_numerical_features(plot_df: pd.DataFrame, feature: str):
    plt.figure(figsize=(9, 6))
    sns.histplot(plot_df[feature], kde=True, color="skyblue")
    plt.title(f'Distribution of {feature}', fontsize=20)
    plt.xlabel(f'{feature}', fontsize=18)  # Increase x-axis label font size
    plt.ylabel('Density', fontsize=18)  # Increase y-axis label font size

    plt.show()


def plot_corr(correlation_matrix: pd.DataFrame):
    # Create a heatmap using Plotly
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",  # Color scale for better distinction
        labels=dict(x="Feature", y="Feature", color="Correlation"),
        x=correlation_matrix.columns,
        y=correlation_matrix.columns)

    # Update layout for clarity
    fig.update_layout(title="Correlation Matrix of Numerical Variables",
                      xaxis_title="Features",
                      yaxis_title="Features",
                      title_x=0.5)  # Center the title

    # Show the plot
    fig.show()


def plot_statistics_by_feature(feature_grouped: pd.DataFrame, categories: list,
                               feature_categories: pd.Series, feature):
    fig = go.Figure()

    # Plot for each gender and metric
    for feature_cat in feature_categories:
        for category in categories:
            fig.add_trace(
                go.Bar(
                    x=[f"{category} Mean", f"{category} Median"],
                    y=feature_grouped[feature_grouped[feature] == feature_cat][
                        (category, 'mean')].tolist() +
                    feature_grouped[feature_grouped[feature] == feature_cat][
                        (category, 'median')].tolist(),
                    name=f"{feature_cat} - {category}",
                    text=feature_grouped[
                        feature_grouped[feature] == feature_cat][
                            (category, 'mean')].round(2).tolist() +
                    feature_grouped[feature_grouped[feature] == feature_cat][
                        (category, 'median')].round(2).tolist(),
                    textposition='auto'))

    # Update the layout
    fig.update_layout(
        title=
        "Grouped Analysis by Gender: Mean and Median of Numerical Variables",
        xaxis_title="Metrics",
        yaxis_title="Values",
        barmode='group',
        title_x=0.5)

    # Show the plot
    fig.show()


def plot_feature_vs_target(outreach_churn_df: pd.DataFrame,
                           churn_categories: list, outreach_outcomes: list):
    # Plot using Plotly
    fig = go.Figure()

    # Add traces for each churn category
    for churn in churn_categories:
        fig.add_trace(
            go.Bar(x=outreach_outcomes,
                   y=outreach_churn_df[int(churn)],
                   name=f"Churn in 30 Days: {churn}",
                   text=outreach_churn_df[int(churn)],
                   textposition='auto'))

    # Update the layout
    fig.update_layout(
        title="Cross-Tabulation: OutReachOutcome vs ChurnIn30Days",
        xaxis_title="OutReach Outcome",
        yaxis_title="Count",
        barmode='group',  # Display bars grouped by outreach outcome
        title_x=0.5  # Center the title
    )

    # Show the plot
    fig.show()


def plot_trend_analysis(df: pd.DataFrame):
    data = df.copy()
    data.set_index('EffectiveDate', inplace=True)

    # Aggregating data by week for mean and median
    weekly_mean_data = data.resample('W-MON').mean()
    weekly_median_data = data.resample('W-MON').median()

    # Now, let's try plotting the trends again with this weekly aggregated data

    # Creating subplots
    weekly_trend_fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        subplot_titles=('Weekly Average App Usage',
                        'Weekly Average Gym Visits in Last 2 Weeks',
                        'Weekly Average Gym Visits in Last 6 Weeks',
                        'Weekly Average Gym Visits in Last 12 Weeks',
                        'Weekly Average Churn in 30 Days'))

    # App Usage Trend
    weekly_trend_fig.add_trace(go.Scatter(x=weekly_mean_data.index,
                                          y=weekly_mean_data['AppUsage'],
                                          mode='lines+markers',
                                          name='Weekly Avg App Usage'),
                               row=1,
                               col=1)

    # Gym Visits in Last 2 Weeks Trend
    weekly_trend_fig.add_trace(go.Scatter(
        x=weekly_mean_data.index,
        y=weekly_mean_data['GymVisitsLast2W'],
        mode='lines+markers',
        name='Weekly Avg Gym Visits Last 2W'),
                               row=2,
                               col=1)

    # Gym Visits in Last 6 Weeks Trend
    weekly_trend_fig.add_trace(go.Scatter(
        x=weekly_mean_data.index,
        y=weekly_mean_data['GymVisitsLast6W'],
        mode='lines+markers',
        name='Weekly Avg Gym Visits Last 6W'),
                               row=3,
                               col=1)

    # Gym Visits in Last 12 Weeks Trend
    weekly_trend_fig.add_trace(go.Scatter(
        x=weekly_mean_data.index,
        y=weekly_mean_data['GymVisitsLast12W'],
        mode='lines+markers',
        name='Weekly Avg Gym Visits Last 12W'),
                               row=4,
                               col=1)

    # Churn in 30 Days Trend
    weekly_trend_fig.add_trace(go.Scatter(x=weekly_mean_data.index,
                                          y=weekly_mean_data['ChurnIn30Days'],
                                          mode='lines+markers',
                                          name='Weekly Avg Churn in 30 Days'),
                               row=5,
                               col=1)

    # Adding traces for median values
    # App Usage Median
    weekly_trend_fig.add_trace(go.Scatter(x=weekly_median_data.index,
                                          y=weekly_median_data['AppUsage'],
                                          mode='lines+markers',
                                          name='Weekly Median App Usage',
                                          line=dict(dash='dash')),
                               row=1,
                               col=1)

    # Gym Visits Last 2W Median
    weekly_trend_fig.add_trace(go.Scatter(
        x=weekly_median_data.index,
        y=weekly_median_data['GymVisitsLast2W'],
        mode='lines+markers',
        name='Weekly Median Gym Visits Last 2W',
        line=dict(dash='dash')),
                               row=2,
                               col=1)

    # Gym Visits Last 6W Median
    weekly_trend_fig.add_trace(go.Scatter(
        x=weekly_median_data.index,
        y=weekly_median_data['GymVisitsLast6W'],
        mode='lines+markers',
        name='Weekly Median Gym Visits Last 6W',
        line=dict(dash='dash')),
                               row=3,
                               col=1)

    # Gym Visits Last 12W Median
    weekly_trend_fig.add_trace(go.Scatter(
        x=weekly_median_data.index,
        y=weekly_median_data['GymVisitsLast12W'],
        mode='lines+markers',
        name='Weekly Median Gym Visits Last 12W',
        line=dict(dash='dash')),
                               row=4,
                               col=1)

    # Churn in 30 Days Median
    weekly_trend_fig.add_trace(go.Scatter(
        x=weekly_median_data.index,
        y=weekly_median_data['ChurnIn30Days'],
        mode='lines+markers',
        name='Weekly Median Churn in 30 Days',
        line=dict(dash='dash')),
                               row=5,
                               col=1)

    # Updating layout
    weekly_trend_fig.update_layout(height=1000,
                                   width=1200,
                                   title_text="Weekly Trend Analysis",
                                   xaxis_title="Date")

    weekly_trend_fig.show()


def detect_and_plot_outliers(df: pd.DataFrame, columns: list):
    data = df.copy()
    # Initialize subplot
    fig = make_subplots(rows=1,
                        cols=len(columns),
                        subplot_titles=[col for col in columns])

    # Initialize a dictionary to hold the outlier counts
    outlier_counts = {}

    # Loop through each column to process
    for i, col in enumerate(columns, start=1):
        # Calculate Z-score
        z_col_name = f"{col}_Z"
        data[z_col_name] = zscore(data[col])

        # Detect outliers using Z-score
        outliers_z = data[abs(data[z_col_name]) > 3]

        # Calculate IQR
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1

        # Detect outliers using IQR method
        outliers_iqr = data[(data[col] < (Q1 - 1.5 * IQR)) |
                            (data[col] > (Q3 + 1.5 * IQR))]

        # Store the count of outliers detected by each method in the dictionary
        outlier_counts[col] = {
            'Z-score method': len(outliers_z),
            'IQR method': len(outliers_iqr)
        }

        # Add box plot for the column
        fig.add_trace(go.Box(y=data[col], name=col), row=1, col=i)

    # Update layout and show figure
    fig.update_layout(height=600,
                      width=1000,
                      title_text="Outlier Detection with Boxplots")
    fig.show()

    # Return the outlier counts dictionary
    return outlier_counts
