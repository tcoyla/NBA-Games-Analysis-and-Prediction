# imports
import numpy as np  # linear algebra
import pandas as pd  # data processing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read data
games_df = pd.read_csv('./data/games.csv')
games_details_df = pd.read_csv('./data/games_details.csv')
teams_df = pd.read_csv('./data/teams.csv')
ranking_df = pd.read_csv('./data/ranking.csv')

# Merge games_df with games_details_df based on 'GAME_ID' to get 'SEASON'
games_details_df = pd.merge(games_df[['SEASON', 'GAME_ID']], games_details_df, on='GAME_ID')

# Calculate Efficiency for each player and each game
games_details_df['EFFICIENCY'] = (games_details_df['PTS'] + games_details_df['REB'] + games_details_df['AST'] + games_details_df['STL'] + games_details_df['BLK'] -
                                  (games_details_df['TO'] + games_details_df['FGA'] - games_details_df['FGM'] + games_details_df['FTA'] - games_details_df['FTM']))

# Keep only players who have played
games_details_df = games_details_df.dropna(subset=['MIN'])

# Keep only efficiency by game and player
player_efficiency_plusMinus = games_details_df[['PLAYER_NAME', 'SEASON', 'EFFICIENCY', 'PLUS_MINUS']].groupby(['PLAYER_NAME', 'SEASON']).mean()

# Group by player name and season, then shift the efficiency for each player
player_efficiency_plusMinus['PREV_EFFICIENCY'] = player_efficiency_plusMinus.groupby('PLAYER_NAME')['EFFICIENCY'].shift(1)

# Group by player name and season, then shift the plus_minus for each player
player_efficiency_plusMinus['PREV_PLUS_MINUS'] = player_efficiency_plusMinus.groupby('PLAYER_NAME')['PLUS_MINUS'].shift(1)

# Reset index
player_efficiency = player_efficiency_plusMinus[['PREV_EFFICIENCY', 'PREV_PLUS_MINUS']].reset_index()

# Drop NaN values resulting from the shift operation
player_efficiency.dropna(inplace=True)

# Merge player efficiency with games details data
merged_data = pd.merge(games_details_df[['TEAM_ID', 'PLAYER_NAME', 'SEASON']], player_efficiency_plusMinus, on=['SEASON', 'PLAYER_NAME'], how='inner')

# Calculate cumulative efficiency and average PLUS_MINUS for each team in each season
cumulative_efficiency = merged_data.groupby(['TEAM_ID', 'SEASON']).sum().reset_index()

cumulative_efficiency = cumulative_efficiency[['TEAM_ID', 'SEASON', 'PREV_EFFICIENCY', 'PREV_PLUS_MINUS']]

# Remove 2003 season for which we have not previous data
cumulative_efficiency = cumulative_efficiency[cumulative_efficiency['SEASON'] != 2003]

# Make a copy of ranking_df
temp = ranking_df.copy()

# Group by 'TEAM_ID' and 'SEASON_ID' and keep only the maximum values of 'G' and 'W'
temp = temp.groupby(['TEAM_ID', 'SEASON_ID', 'W_PCT'])[['G', 'W', 'CONFERENCE']].max().reset_index()

# Extract last four digits if SEASON_ID exceeds 10,000
temp['SEASON_ID'] = temp['SEASON_ID'] % 10000

# Remove rows where 'G' column is not equal to 82
temp = temp[temp['G'] == 82]

# Rename column
temp = temp.rename(columns={'SEASON_ID': 'SEASON'})

# Shift the 'W_PCT' column by one season to get previous season's winning percentage
temp['PREV_W_PCT'] = temp.groupby('TEAM_ID')['W_PCT'].shift(1)

# Create the dataframe for machine learning application
ml_games_df = pd.merge(cumulative_efficiency, temp, on=['TEAM_ID', 'SEASON'])

# Rename the 'W' column to 'WINS'
ml_games_df = ml_games_df.rename(columns={'W': 'WINS'})

# Replace TEAM_ID to TEAM_NAME using teams_df
ml_games_df['TEAM_NAME'] = ml_games_df['TEAM_ID'].map(teams_df.set_index('TEAM_ID')['NICKNAME'])

# Arrange columns
ml_games_df = ml_games_df[['TEAM_NAME', 'SEASON', 'PREV_EFFICIENCY', 'PREV_PLUS_MINUS', 'CONFERENCE', 'WINS']].reset_index(drop=True)

# Function to get prediction results for a given season
def predict_winsBySeason(season):
    # Preparing training dataset for predictions
    ml_games_train = ml_games_df.loc[(ml_games_df['SEASON'] == season-1)]

    # Copy features and outcome columns
    ml_games_train_features = ml_games_train[['PREV_EFFICIENCY', 'PREV_PLUS_MINUS', 'CONFERENCE']].copy()
    ml_games_train_outcome = ml_games_train[['WINS', 'CONFERENCE']].copy()

    # Define X_train and y_train
    X_train, y_train = ml_games_train_features[['PREV_EFFICIENCY', 'PREV_PLUS_MINUS']], ml_games_train_outcome['WINS']

    # Preparing testing dataset
    ml_games_test = ml_games_df.loc[ml_games_df['SEASON'] == season]

    ml_games_test_features = ml_games_test.copy()[['PREV_EFFICIENCY', 'PREV_PLUS_MINUS', 'CONFERENCE']]
    ml_games_test_outcome = ml_games_test.copy()[['WINS', 'CONFERENCE']]

    X_test, y_test =  ml_games_test_features[['PREV_EFFICIENCY', 'PREV_PLUS_MINUS']], ml_games_test_outcome['WINS']

    # Initialize the regression model with the current alpha value
    model = LinearRegression()
    
    # Train the machine learning model
    model.fit(X_train, y_train)
    
    # Predict the wins for the 2018 season
    y_pred = model.predict(X_test).astype(int)
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Create a DataFrame to store the results
    ml_results_df = pd.DataFrame({'Team Name': ml_games_test['TEAM_NAME'], 'Actual Wins': y_test, 'Predicted Wins': y_pred, 'Conference': ml_games_test['CONFERENCE']})

    return ml_results_df

# Function to predict wins for all the season
def predict_allWins():
    # Store all the prediction results
    results_list = []
    # Get results for all the possible seasons 
    for season in range(2006, 2019):
        # 2011 season is incomplete so we remove it
        if((season != 2011) & (season != 2012)):
            results_list.append(predict_winsBySeason(season))
    
    return results_list
