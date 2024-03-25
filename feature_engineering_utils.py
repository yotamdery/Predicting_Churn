#########################################################
# THIS SCRIPT IS TO CONTAIN FEATURE ENGINEERING FUNCTIONS
#########################################################

# Imports
import pandas as pd
from datetime import timedelta
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler


def calc_bmi_categories(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df['Height_m'] = df['Height'] / 100  # Convert height from cm to m
    df['BMI'] = df['Weight'] / (df['Height_m']**2)
    # Define BMI bins and labels
    bmi_bins = [0, 18.5, 25, 30, float('inf')]
    bmi_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obese']

    # Categorize BMI values into bins
    df['BMI_Category'] = pd.cut(df['BMI'],
                                bins=bmi_bins,
                                labels=bmi_labels,
                                right=False)
    df = df.drop(columns=['BMI', 'Height_m'], axis=1)
    return df


def calc_weight_height_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df['WeightHeightRatio'] = df['Weight'] / df['Height']
    return df


def create_time_series_splits(df: pd.DataFrame) -> List[Tuple]:
    # Determine the range of your dataset
    start_date = df['EffectiveDate'].min()
    end_date = df['EffectiveDate'].max()

    # Initialize the start and end of the first training period
    train_start = start_date
    train_end = start_date + timedelta(weeks=46)

    # Placeholder for your splits - each element is a tuple
    splits = []
    # Generate the time series splits
    while train_end < end_date:
        test_end = train_end + timedelta(weeks=1)
        train_indices = df[(df['EffectiveDate'] >= train_start)
                           & (df['EffectiveDate'] < train_end)].index
        test_indices = df[(df['EffectiveDate'] >= train_end)
                          & (df['EffectiveDate'] < test_end)].index

        splits.append((train_indices, test_indices))

        # Move to the next period
        train_end = test_end

    return splits


# def calc_AppUsage_median_per_week(data: pd.DataFrame) -> pd.Series:
#     df = data.copy()
#     # Set the date to the beginning of the week for each entry
#     df['Week'] = df['EffectiveDate'] - pd.to_timedelta(
#         df['EffectiveDate'].dt.dayofweek, unit='d')
#     series_res = df.groupby('Week')['AppUsage'].median().reset_index(
#         name='MedianAppUsage')
#     return series_res


def calc_diff_median_app_usage_per_user_per_week(
        data: pd.DataFrame) -> pd.DataFrame:

    data['Week'] = data['EffectiveDate'] - pd.to_timedelta(
        data['EffectiveDate'].dt.dayofweek, unit='d')

    weekly_median_app_usage = data.groupby(
        'Week')['AppUsage'].median().reset_index(name='MedianAppUsage')

    # Merge the median values back with the original dataset
    data_with_median = data.merge(weekly_median_app_usage,
                                  on='Week',
                                  how='inner')

    # Calculate the difference between each member's 'AppUsage' and the weekly median
    data_with_median['AppUsageDiffFromMedian'] = data_with_median[
        'AppUsage'] - data_with_median['MedianAppUsage']

    data_with_median = data_with_median.drop(
        columns=['Week', 'MedianAppUsage'], axis=1)
    return data_with_median


def create_age_bins(df: pd.DataFrame) -> pd.DataFrame:
    min_age = df['Age'].min()
    max_age = df['Age'].max()

    # Create age bins from the minimum to the maximum age, with a step of 5
    bins = list(range(min_age - min_age % 5, max_age + 5 - max_age % 5, 5))

    # Labels for the age groups
    labels = [f'{i}-{i + 4}' for i in bins[:-1]]

    # Categorize ages into bins
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
    df = df.drop(columns='Age', axis=1)
    return df


def calc_visit_change(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate the change in gym visit frequency from the last 2 weeks to the last 6 weeks
    df['GymVisitDiff_2W_to_6W'] = df['GymVisitsLast6W'] - (
        df['GymVisitsLast2W'])
    # Calculate the change in gym visit frequency from the last 6 weeks to the last 12 weeks
    df['GymVisitDiff_6W_to_12W'] = df['GymVisitsLast12W'] - (
        df['GymVisitsLast6W'])

    epsilon = 1e-6
    # Calculate the change in gym visit frequency from the last 2 weeks to the last 6 weeks
    df['GymVisitRatio_2W_to_6W'] = df['GymVisitsLast2W'] / (
        df['GymVisitsLast6W'] + epsilon)
    # Calculate the change in gym visit frequency from the last 6 weeks to the last 12 weeks
    df['GymVisitRatio_6W_to_12W'] = df['GymVisitsLast6W'] / (
        df['GymVisitsLast12W'] + epsilon)

    return df


def calc_statistics_and_norm_object_engagement_score(data: pd.DataFrame):
    df = data.copy()

    # Aggregate data per member by averaging the engagement-related metrics
    member_aggregated = df.groupby('MemberID')[[
        'AppUsage', 'GymVisitRatio_2W_to_6W', 'GymVisitRatio_6W_to_12W',
        'GymVisitDiff_2W_to_6W', 'GymVisitDiff_6W_to_12W'
    ]].mean().reset_index()

    # Normalize the components
    scaler = MinMaxScaler()
    scaler.fit_transform(member_aggregated[[
        'AppUsage', 'GymVisitRatio_2W_to_6W', 'GymVisitRatio_6W_to_12W',
        'GymVisitDiff_2W_to_6W', 'GymVisitDiff_6W_to_12W'
    ]])

    return scaler


def calc_engagement_score(data: pd.DataFrame,
                          member_aggregated: pd.DataFrame = None,
                          scaler: MinMaxScaler = None) -> pd.DataFrame:
    df = data.copy()
    # Calculate mean values if not provided (for X_train)
    member_aggregated = df.groupby('MemberID')[[
        'AppUsage', 'GymVisitRatio_2W_to_6W', 'GymVisitRatio_6W_to_12W',
        'GymVisitDiff_2W_to_6W', 'GymVisitDiff_6W_to_12W'
    ]].mean().reset_index()

    if scaler is None:
        # Fit scaler if not provided (for X_train)
        scaler = MinMaxScaler()
        components_aggregated_normalized = scaler.fit_transform(
            member_aggregated.iloc[:, 1:])
    else:
        # Use provided scaler to transform data (for X_test)
        components_aggregated_normalized = scaler.transform(
            member_aggregated.iloc[:, 1:])
    # Calculate the Engagement Score as the mean of the normalized components
    member_aggregated[
        'EngagementScore'] = components_aggregated_normalized.mean(axis=1)
    member_aggregated['EngagementScore'] = member_aggregated['EngagementScore'].clip(lower=0)

    # Categorize the Engagement Score into 'Low', 'Medium', 'High'
    score_bins = [0, 1 / 3, 2 / 3, 1]
    score_labels = ['Low', 'Medium', 'High']
    member_aggregated['EngagementCategory'] = pd.cut(
        member_aggregated['EngagementScore'],
        bins=score_bins,
        labels=score_labels,
        include_lowest=True)

    # Merge the aggregated engagement data back into the original DataFrame
    # This retains all original features and adds the 'EngagementScore' and 'EngagementCategory'
    # We perform inner join as some members in the test set have no samples in the training set
    merged_df = pd.merge(df,
                         member_aggregated[['MemberID', 'EngagementCategory']],
                         on='MemberID',
                         how='inner')

    return merged_df


def create_categorical_encoding(data: pd.DataFrame) -> pd.DataFrame:
    # Disatvantage: get_dummies on all data assumes that the categories are the same for every week/run, and this is not happening always after train/test split
    # I assume fixed data for the simplicity. in real world applications, I'd probably need to do the train/test split before the get dummies.
    data= pd.get_dummies(data,
                        columns=['BMI_Category', 'DEXAScanResult', 'OutReachOutcome'],
                        drop_first=True)
    # Define the mapping that reflects the order of the categories
    engagement_mapping = {'Low': 0, 'Medium': 1, 'High': 2}

    # Apply this mapping to the 'EngagementCategory' column
    data['EngagementCategory'] = data['EngagementCategory'].map(engagement_mapping).astype(int)
    return data