#########################################################
# THIS SCRIPT IS TO CONTAIN FEATURE ENGINEERING FUNCTIONS
#########################################################

# Imports
import pandas as pd
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

    print("We use both difference and ratio methods.\n"
    "Diff: directly shows the increase or decrease in gym visits, making it easy to identify trends or sudden changes in behavior.\n"
    "Ratio: The ratio provides a sense of proportion, showing how gym visits have changed relative to a previous period. \n This can be more meaningful when comparing across members with different levels of overall activity.")
    return df


def calc_engagement_score(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # Aggregate data per member by averaging the engagement-related metrics
    member_aggregated = df.groupby('MemberID')[[
        'AppUsage', 'GymVisitRatio_2W_to_6W', 'GymVisitRatio_6W_to_12W',
        'GymVisitDiff_2W_to_6W', 'GymVisitDiff_6W_to_12W'
    ]].mean().reset_index()

    # Normalize the components
    scaler = MinMaxScaler()
    components_aggregated_normalized = scaler.fit_transform(member_aggregated[[
        'AppUsage', 'GymVisitRatio_2W_to_6W', 'GymVisitRatio_6W_to_12W',
        'GymVisitDiff_2W_to_6W', 'GymVisitDiff_6W_to_12W'
    ]])

    # Calculate the Engagement Score as the mean of the normalized components
    member_aggregated[
        'EngagementScore'] = components_aggregated_normalized.mean(axis=1)

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
    merged_df = pd.merge(df,
                         member_aggregated[['MemberID', 'EngagementCategory']],
                         on='MemberID',
                         how='left')

    print(
        "Note: The aggregation is performed to each member, buy the normalization is done across all members for each metric!"
    )
    return merged_df


def create_categorical_encoding(data: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(data,
                        columns=['DEXAScanResult', 'OutReachOutcome'],
                        drop_first=True)
    # Define the mapping that reflects the order of the categories
    engagement_mapping = {'Low': 0, 'Medium': 1, 'High': 2}

    # Apply this mapping to the 'EngagementCategory' column
    df['EngagementCategory'] = df['EngagementCategory'].map(engagement_mapping)
    print("Note: We use one-hot encoding and ordinal encdoing in this method")
    return df
