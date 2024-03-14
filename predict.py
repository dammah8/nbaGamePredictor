import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Read the dataset
nba_df = pd.read_csv("nba_games.csv", index_col=0)

# Preprocessing
nba_df = nba_df.sort_values("date")
nba_df = nba_df.reset_index(drop=True)
del nba_df["mp.1"]
del nba_df["mp_opp.1"]
del nba_df["index_opp"]

# Function to add target column
def add_target_column(group):
    group["target"] = group["won"].shift(-1)
    return group

# Group by team and apply target column addition
nba_df = nba_df.groupby("team", group_keys=False).apply(add_target_column)

# Handling missing values in the target column
nba_df["target"][pd.isnull(nba_df["target"])] = 2
nba_df["target"] = nba_df["target"].astype(int, errors="ignore")

# Split data into training and testing sets
rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)
sfs = SequentialFeatureSelector(rr, 
                                n_features_to_select=30, 
                                direction="forward",
                                cv=split,
                                n_jobs=1
                               )

# Define removed columns and selected columns for feature selection
removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = nba_df.columns[~nba_df.columns.isin(removed_columns)]

# Scale selected columns
scaler = MinMaxScaler()
nba_df[selected_columns] = scaler.fit_transform(nba_df[selected_columns])

# Perform feature selection
sfs.fit(nba_df[selected_columns], nba_df["target"])
selected_predictors = list(selected_columns[sfs.get_support()])

# Function to perform backtesting
def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []
    seasons = sorted(data["season"].unique())
    
    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data[data["season"] == season]
        
        model.fit(train[predictors], train["target"])
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)

# Perform backtesting
predictions = backtest(nba_df, rr, selected_predictors)

# Calculate and print accuracy score
accuracy = accuracy_score(predictions["actual"], predictions["prediction"])
print("Accuracy Score:", accuracy)
