# import useful librairies
import pandas as pd 
from dash import Dash, html, dcc, Output, Input
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import base64
from itertools import cycle

import dash_prediction

# Read data
games_df = pd.read_csv('./data/games.csv')
games_details_df = pd.read_csv('./data/games_details.csv')
teams_df = pd.read_csv('./data/teams.csv')
ranking_df = pd.read_csv('./data/ranking.csv')

# Remove 2022 season which is not complete
games_df = games_df[games_df['SEASON'] != 2022]

# Merge games_df with games_details_df based on 'GAME_ID' to get 'SEASON'
games_details_df = pd.merge(games_df[['SEASON', 'GAME_ID', 'GAME_DATE_EST']], games_details_df, on='GAME_ID')

# Convert 'GAME_DATE_EST' column to datetime format
games_details_df['GAME_DATE_EST'] = pd.to_datetime(games_details_df['GAME_DATE_EST'])
games_df['GAME_DATE_EST'] = pd.to_datetime(games_df['GAME_DATE_EST'])

# Filter out games with date between April 16 and october for each season
games_details_df = games_details_df[((games_details_df['GAME_DATE_EST'].dt.month == 4) & (games_details_df['GAME_DATE_EST'].dt.day <= 16)) 
                                    | (games_details_df['GAME_DATE_EST'].dt.month >= 9) | (games_details_df['GAME_DATE_EST'].dt.month < 4)]
games_df = games_df[((games_df['GAME_DATE_EST'].dt.month == 4) & (games_df['GAME_DATE_EST'].dt.day <= 16)) | (games_df['GAME_DATE_EST'].dt.month >= 9) 
                    | (games_df['GAME_DATE_EST'].dt.month < 4)]

games_details_df = games_details_df.drop('GAME_DATE_EST', axis=1)

# Keep only players who have played
games_details_df = games_details_df.dropna(subset=['MIN'])

# Select variables
teams_df = teams_df[["TEAM_ID", "CITY", "NICKNAME", "ARENACAPACITY"]]
games_df = games_df.drop(["GAME_STATUS_TEXT", "GAME_DATE_EST"], axis=1)

# Remove NAN values
games_df = games_df.dropna()

# Create home games dataframe
home_games_df = games_df[["SEASON", "TEAM_ID_home", "HOME_TEAM_WINS", "PTS_home", "AST_home", "REB_home", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home"]]
home_games_df = pd.merge(home_games_df, teams_df, left_on = "TEAM_ID_home", right_on = "TEAM_ID", how = "inner")
home_games_df = home_games_df.rename(columns={"ARENACAPACITY" : "ARENA_CAPACITY", "HOME_TEAM_WINS" : "WIN", "PTS_home" : "PTS", "AST_home" : "AST",
                                              "REB_home" : "REB", "FG_PCT_home" : "FG_PCT", "FT_PCT_home" : "FT_PCT", "FG3_PCT_home" : "FG3_PCT"})
home_games_df = home_games_df.drop(["TEAM_ID","TEAM_ID_home"], axis=1)

# Create away games dataframe
away_games_df = games_df[["SEASON", "TEAM_ID_away", "HOME_TEAM_WINS", "PTS_away", "AST_away", "REB_away", "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away"]]
away_games_df = pd.merge(away_games_df, teams_df, left_on = "TEAM_ID_away", right_on = "TEAM_ID", how = "inner")
away_games_df["HOME_TEAM_WINS"] = 1 - away_games_df["HOME_TEAM_WINS"]
away_games_df = away_games_df.rename(columns={"ARENACAPACITY" : "ARENA_CAPACITY", "HOME_TEAM_WINS" : "WIN", "PTS_away" : "PTS", "AST_away" : "AST",
                                               "REB_away" : "REB", "FG_PCT_away" : "FG_PCT", "FT_PCT_away" : "FT_PCT", "FG3_PCT_away" : "FG3_PCT"})
away_games_df = away_games_df.drop(["TEAM_ID","TEAM_ID_away"], axis=1)

# Concatenate home and away games dataframes
all_games_df = pd.concat([home_games_df, away_games_df], ignore_index=True)

# Create a dataframe by season and team
season_team_df = all_games_df[['SEASON', 'NICKNAME', 'WIN', 'PTS', 'AST', 'REB', 'FG_PCT', 'FT_PCT', 'FG3_PCT']].groupby(['SEASON', 'NICKNAME']).mean().reset_index()

# Get the total number of shoot for each player
selected_players_df = games_details_df[['PLAYER_NAME', 'FGA', 'FTA', 'FG3A']].groupby('PLAYER_NAME').sum()

# Remove players who have pratically never shoot
players_names = selected_players_df.loc[(selected_players_df['FGA'] > 50) & (selected_players_df['FTA'] > 30)
                                           & (selected_players_df['FG3A'] > 10)].index.tolist()
# Get the players stats
players_stats_df = games_details_df[['TEAM_ID', 'SEASON', 'PLAYER_NAME', 'PTS', 'AST', 'REB', 'STL', 'BLK', 'TO', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'FTA', 'FGA', 'FG3A'
                                    ]].loc[games_details_df['PLAYER_NAME'].isin(players_names)]

# Merge the players_stats_df DataFrame with the teams_df DataFrame on the 'TEAM_ID' column
players_stats_df = players_stats_df.merge(teams_df[['TEAM_ID', 'NICKNAME']], on='TEAM_ID', how='left')
# Rename the 'NICKNAME' column to 'TEAM_NAME' if necessary
players_stats_df.rename(columns={'NICKNAME': 'TEAM_NAME'}, inplace=True)

# Add a column to evaluate player efficiency
players_stats_df['EFFICIENCY'] = (players_stats_df['PTS'] + players_stats_df['REB'] + players_stats_df['AST'] + players_stats_df['STL'] + players_stats_df['BLK']
                                  - (players_stats_df['TO'] + players_stats_df['FGA'] * (1 - players_stats_df['FG_PCT']) + 
                                     players_stats_df['FTA'] * (1 - players_stats_df['FT_PCT'])))

# Calculate average cumul_efficiency by player
cumul_efficiency_df = players_stats_df.groupby(['PLAYER_NAME', 'TEAM_NAME', 'SEASON']).mean().dropna()

# Create players statistics average dataframe
players_stats_df = players_stats_df.groupby(['PLAYER_NAME', 'TEAM_NAME', 'SEASON']).mean().dropna()

# Define Teams options 
team_options = [{'label': 'All Teams', 'value': 'All Teams'}] + [{'label': team, 'value': team} for team in home_games_df['NICKNAME'].sort_values().unique()]

# Feature options for feature dropdown 
feature_options = {
    'All Teams': [
        {'label': 'Arena Capacity', 'value': 'ARENA_CAPACITY'},
        {'label': 'Win Percentage', 'value': 'WIN'},
        {'label': 'Points', 'value': 'PTS'},
        {'label': 'Assists', 'value': 'AST'},
        {'label': 'Rebounds', 'value': 'REB'},
        {'label': 'Field Goal Percentage', 'value': 'FG_PCT'},
        {'label': 'Free Throw Percentage', 'value': 'FT_PCT'},
        {'label': 'Field Goal at 3 Point Percentage', 'value': 'FG3_PCT'}
    ],
    'Other': [
        {'label': 'Team Stats', 'value': 'team_stats'},
        {'label': 'Players Stats', 'value': 'player_stats'},
    ]
}

# Define year options for year dropdown
year_options = [
    {'label': 'Season 2006 / 2007', 'value': 2006},
    {'label': 'Season 2007 / 2008', 'value': 2007},
    {'label': 'Season 2008 / 2009', 'value': 2008},
    {'label': 'Season 2009 / 2010', 'value': 2009},
    {'label': 'Season 2010 / 2011', 'value': 2010},
    {'label': 'Season 2013 / 2014', 'value': 2013},
    {'label': 'Season 2014 / 2015', 'value': 2014},
    {'label': 'Season 2015 / 2016', 'value': 2015},
    {'label': 'Season 2016 / 2017', 'value': 2016},
    {'label': 'Season 2017 / 2018', 'value': 2017},
    {'label': 'Season 2018 / 2019', 'value': 2018}
]

# Define colors for Teams
team_colors = {
    'All Teams': ['#006BB6', '#ED174C'],
    '76ers': ['#ED174C', '#006BB6'],
    'Bucks': ['#EEE1C6', '#00471B'],
    'Bulls': ['#CE1141', '#000000'],
    'Cavaliers': ['#FFB81C', '#6F263D'],
    'Celtics': ['#BA9653', '#007A33'],
    'Clippers': ['#C8102E', '#1D428A'],
    'Grizzlies': ['#5D76A9', '#12173F'],
    'Hawks': ['#C1D32F', '#E03A3E'],
    'Heat': ['#F9A01B', '#98002E'],
    'Hornets': ['#00788C', '#1D1160'],
    'Jazz': ['#F9A01B', '#002B5C'],
    'Kings': ['#63727A', '#5A2D81'],
    'Knicks': ['#F58426', '#006BB6'],
    'Lakers': ['#FDB927', '#552583'],
    'Magic': ['#C4CED4', '#0077C0'],
    'Mavericks': ['#00538C', '#002B5E'],
    'Nets': ['#D3D3D3', '#000000'],
    'Nuggets': ['#FEC524', '#0E2240'],
    'Pacers': ['#FDBB30', '#002D62'],
    'Pelicans': ['#85714D', '#0C2340'],
    'Pistons': ['#C8102E', '#1D428A'],
    'Raptors': ['#CE1141', '#000000'],
    'Rockets': ['#CE1141', '#000000'],
    'Spurs': ['#C4CED4', '#000000'],
    'Suns': ['#E56020', '#1D1160'],
    'Thunder': ['#007AC1', '#EF3B24'],
    'Timberwolves': ['#78BE20', '#0C2340'],
    'Trail Blazers': ['#E03A3E', '#000000'],
    'Warriors': ['#FFC72C', '#1D428A'],
    'Wizards': ['#E31837', '#002B5C']
}

# Function to load an image in the dashboard
def load_image(path, width, height):
    with open(path, 'rb') as f:
        image_binary = f.read()

    # Encode the binary data as base64
    image_base64 = base64.b64encode(image_binary).decode('utf-8')

    # Use the base64-encoded string as the src attribute for the html.Img component
    return html.Img(src=f'data:image/png;base64,{image_base64}', width=width, height=height, style={'display': 'block', 'margin-right': '50px'})

# Initialize the app with a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# App layout
data_visualization_layout = dbc.Container([
    html.H1(id='title', className='text-center mt-5', style={'fontSize': '45px', 'textShadow': '1px 1px 2px white'}),

    dbc.Row([
        dbc.Col(
            load_image('./data/logos/All Teams.png',width=350, height=310),
            width = 4,
            style={'marginLeft': '100px', 'marginBottom' : '50px'},
            id='image-container'
        )
    ,   
        dbc.Col([
            html.H4("Choose a team", className='text-center mb-0', style={'color' : 'grey'}),
            dcc.Dropdown(
                options=team_options,
                value='All Teams',
                id='team-dropdown',
                clearable=False,
                className='mb-3'),
            ],
            style={'width': '4', 'marginTop': '120px', 'marginLeft': '-150px'})
        ,
        dbc.Col(
            [
                html.H4("Choose a feature to visualize", className='text-center mb-0', style={'color' : 'grey'}),
                dcc.Dropdown(    
                    value='ARENA_CAPACITY',              
                    id='feature-dropdown',
                    className='mb-3',
                    clearable=False
                ),
            ],
            align="center",
            style={'width': '4', 'marginBottom': '80px', 'margin-left': '200px', 'margin-right': '300px'},
        ),
    ],
    style={'height': '330px'}
    ),

    dbc.Row([
        dbc.Col(
            html.Div(id='top-left-graph'),
            width=4,
            align="center",
        )
    ,
        dbc.Col(
            dcc.Graph(id='top-center-graph'),
            width=4,
            align="center",
        )
    ,
        dbc.Col(
            dcc.Graph(id='top-right-graph'),
            width=4,
            align="center",
        )
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='bottom-graph'),
            width=15,
        )
]),
], fluid=True)

# Define layout for the prediction page
prediction_layout = dbc.Container([
    html.H1(id='prediction-title', className='text-center mt-5', style={'color': '#333333', 'fontSize': '45px', 'textShadow': '1px 1px 2px white'}),
    dbc.Row([
        dbc.Col(
            load_image('./data/logos/west.png', width=330, height=310),
            style={'marginLeft': '250px', 'marginBottom' : '100px'},
        ),
        dbc.Col([
            html.H4("Choose a season", className='text-center mb-0', style={'color' : 'grey'}),
            dcc.Dropdown(
                options=year_options,
                value='2018',
                id='prediction-year-dropdown',
                clearable=False,
                className='mb-3'),
            ],
            width = 3,
            style={'marginTop': '80px', 'marginLeft': '-190px', 'marginRight': '50px'}
        ),
        dbc.Col(
            load_image('./data/logos/east.png', width=340, height=310),
            style={'marginBottom' : '100px', 'marginRight': '10px'},
        )
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Choose a prediction", className='text-center mb-0', style={'color' : 'grey'}),
            dcc.Dropdown(
                id='prediction-dropdown',
                options=[
                    {'label': 'Wins', 'value': 'wins'},
                    {'label': 'Rank', 'value': 'rank'}
                ],
                value='wins',
                clearable=False,
                className='mb-3'),
        ],
            width=3,
            style={'marginTop': '-220px', 'marginLeft': '687px'}
        ),
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='prediction-left-graph'),
            width=6,
            align="center",
            style={'marginTop' : '-80px'}
        ),
        dbc.Col(
            dcc.Graph(id='prediction-right-graph'),
            width=6,
            align="center",
            style={'marginTop' : '-80px'}
        )
    ])
], fluid=True)

# Define app layout with navigation bar and page content
app.layout = html.Div([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Visualization Page", href="/", style={'color': 'black', 'fontSize': '13px'})),
            dbc.NavItem(dbc.NavLink("Prediction Page", href="/prediction", style={'color': 'black', 'fontSize': '13px'})),
        ],
        brand=html.Span("NBA Dashboard From 2003 to 2021", style={'color': 'red', 'fontSize': '50px', 'fontWeight': 'bold', 'marginLeft': '500px'}),
        brand_href="/",
        color="light",
        dark=True,
        fluid=True
    ),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Define callback to switch between pages based on URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)

def display_page(pathname):
    if pathname == '/prediction':
        return prediction_layout
    else:
        return data_visualization_layout
    
    
######################################################################################################################################################################
                                                               # Data Visualization #
######################################################################################################################################################################


# Define callback to update the title
@app.callback(
    Output('title', 'children'),
    [Input('team-dropdown', 'value')]
)
def update_title(chosen_team):
    if (chosen_team == 'All Teams'):
        color = '#006BB6'
        # Update the title text based on the chosen team
        title_text = 'Data Visualization'
    
    else:
        # Determine the color based on the chosen team
        color = team_colors[chosen_team][1]
        # Update the title text based on the chosen team
        title_text = f'{chosen_team}'

    return html.Span(title_text, style={'color': color})

# Callback to update the options of the feature dropdown based on the team selection
@app.callback(
    [Output('feature-dropdown', 'options'),
     Output('feature-dropdown', 'value')],
    Input('team-dropdown', 'value')
)
def update_feature_options(selected_team):
    default_value = 'WIN'
    if selected_team != 'All Teams':
        selected_team = 'Other'
        default_value = 'team_stats'
    return [feature_options[selected_team], default_value]

# Define a callback to update the image based on team dropdown selection
@app.callback(
    Output('image-container', 'children'),
    Input('team-dropdown', 'value')
)
def update_image(selected_team):
    # Get the logo image path of the selected team
    image_path = f'./data/logos/{selected_team}.png'
    # Load the image
    image_component = load_image(image_path, width=400, height=350)

    return image_component


# Define a callback to update the pie chart based on feature dropdown selection
@app.callback(
    Output('top-left-graph', 'children'),
    [Input('feature-dropdown', 'value'),
     Input('team-dropdown', 'value')]
)
def update_top_left(chosen_col, chosen_team):
    if (chosen_team == 'All Teams'):
        if (chosen_col == 'ARENA_CAPACITY'):
            # Get the corresponding labels and values from the DataFrame
            values = home_games_df[[chosen_col, 'NICKNAME']].loc[home_games_df['ARENA_CAPACITY'] > 0].groupby('NICKNAME').mean()
            total_capacity = values.sum()
            values = (values * (100 / total_capacity))[chosen_col].tolist()
            labels = home_games_df['NICKNAME'].loc[home_games_df['ARENA_CAPACITY']>0].unique()
            # Create pie chart trace
            pie_chart = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=['blue','red']))])
            pie_chart.update_layout(title=f'{chosen_col} Distribution Through All Teams', title_x=0.5, title_y=0.9, title_font_size=24)
            
        else:
            # Get the corresponding values from the DataFrame
            values = 100 * [home_games_df[chosen_col].mean(), away_games_df[chosen_col].mean()]
            labels = ['Home', 'Away']
            # Create pie chart trace
            pie_chart = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=['blue','red']))])
            # Update layout
            pie_chart.update_layout(title=f'{chosen_col} Distribution', title_x=0.5, title_y=0.9, title_font_size=24,legend_title='Game Location',
                                    legend=dict(orientation='h', yanchor='bottom', y=-0.2, x=0.2))
    else:
        if (chosen_col == 'team_stats'):
            # Get the corresponding labels and values from the DataFrame
            value = ((home_games_df[['WIN', 'NICKNAME']].loc[home_games_df['NICKNAME'] == chosen_team].groupby('NICKNAME').mean() + 
                      away_games_df[['WIN', 'NICKNAME']].loc[away_games_df['NICKNAME'] == chosen_team].groupby('NICKNAME').mean()) / 2)['WIN'].values[0]
            values = [value, 1 - value]
            labels = ['Win', 'Loss']
            # Create pie chart trace
            pie_chart = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=team_colors[chosen_team]))])
            pie_chart.update_layout(title='Percentage of Win', title_x=0.5, title_y=0.9, title_font_size=24, 
                                    legend=dict(orientation='h', yanchor='bottom', y=-0.2, x=0.33)
            )
            
        else:
            # Filter data for the chosen team
            team_players_df = players_stats_df.loc[(players_stats_df.index.get_level_values('TEAM_NAME') == chosen_team) &
                                                    (players_stats_df.index.get_level_values('SEASON') == 2018)].sort_values(by='EFFICIENCY').tail(5)

            # Calculate total points scored by the team
            total_efficiency = team_players_df['EFFICIENCY'].sum()

            # Calculate percentage contribution of each player
            team_players_df['Contribution (%)'] = (team_players_df['EFFICIENCY'] / total_efficiency) * 100

            values = team_players_df['Contribution (%)'].tolist()
            labels = team_players_df.index.get_level_values('PLAYER_NAME')
            # Create pie chart trace
            pie_chart = go.Figure(data=[go.Pie(labels=labels, values=values)])
            pie_chart.update_layout(title='Percentage of Efficiency by Top 5 Players in 2018', title_x=0.5, title_y=0.9, title_font_size=24, 
                                    legend=dict(orientation='h', yanchor='bottom', y=-0.3, x=0.15)
            )

    # Create pie chart component
    pie_chart_component = dcc.Graph(id='pie-chart-graph', figure=pie_chart)

    return pie_chart_component

# Define a callback to update the pie chart based on feature dropdown selection
@app.callback(
    Output('top-center-graph', 'figure'),
    [Input('feature-dropdown', 'value'),
     Input('team-dropdown', 'value')]
)
def update_top_center(chosen_col, chosen_team):
    # Create a figure
    fig = go.Figure()

    # Teams colors for graph
    team_color_palette = cycle(team_colors[chosen_team])

    if (chosen_team == 'All Teams'):

        if (chosen_col == 'ARENA_CAPACITY'):
            # Sort teams based on ARENA_CAPACITY
            arena_capacity_df = home_games_df.loc[home_games_df["ARENA_CAPACITY"]>0].groupby('NICKNAME')[['ARENA_CAPACITY','WIN']].mean().reset_index()
            sorted_arenaCapacity = arena_capacity_df.sort_values(by='ARENA_CAPACITY').tail(10)

            # Define colors of the graph
            graph_colors = [next(team_color_palette) for _ in range(10)]

            # Add trace
            fig.add_trace(go.Bar(x=sorted_arenaCapacity['NICKNAME'], y=sorted_arenaCapacity['ARENA_CAPACITY'], marker_color=graph_colors))
            
            # Update layout
            fig.update_layout(title='First Teams With Highest Arena Capacity',
                            xaxis_title='Team',
                            yaxis_title='Arena Capacity',
                            title_x=0.5,
                            title_y=0.9,
                            title_font_size=24,
                            xaxis_tickangle=-45)
            
        elif (chosen_col == 'WIN'):
             # Calculate points difference for home team wins
            home_wins_games = games_df.loc[games_df['HOME_TEAM_WINS'] == 1][['PTS_home', 'PTS_away']]
            home_wins_games['PTS_difference'] = home_wins_games['PTS_home'] - home_wins_games['PTS_away']

            # Categorize points difference into three levels
            home_wins_games['Difference_Category'] = pd.cut(home_wins_games['PTS_difference'], bins=[-200, 10, 20, 200], labels=['Under 10', '10-20', 'Over 20'])

            # Calculate points difference for away team wins
            away_wins_games = games_df.loc[games_df['HOME_TEAM_WINS'] == 0][['PTS_home', 'PTS_away']]
            away_wins_games['PTS_difference'] = away_wins_games['PTS_away'] - away_wins_games['PTS_home']

            # Categorize points difference into three levels
            away_wins_games['Difference_Category'] = pd.cut(away_wins_games['PTS_difference'], bins=[-200, 10, 20, 200], labels=['Under 10', '10-20', 'Over 20'])

            # Count the number of home team wins for each category
            home_wins_count_by_category = home_wins_games['Difference_Category'].value_counts().sort_index()
            
            # Count the number of away team wins for each category
            away_wins_count_by_category = away_wins_games['Difference_Category'].value_counts().sort_index()

            # Calculate percentage of home team wins for each category
            home_wins_percentage_by_category = (home_wins_count_by_category / (away_wins_count_by_category.sum() + home_wins_count_by_category.sum())) * 100

            # Calculate percentage of away team wins for each category
            away_wins_percentage_by_category = (away_wins_count_by_category / (away_wins_count_by_category.sum() + home_wins_count_by_category.sum())) * 100

            # Create a figure
            fig = go.Figure()

            # Add traces for home team wins
            fig.add_trace(go.Bar(
                x=home_wins_count_by_category.index,
                y=home_wins_count_by_category.values,
                name='Home Team Wins',
                marker_color='blue',
                text=home_wins_percentage_by_category.round(2).astype(str) + '%',
                textposition='auto'
            ))

            # Add traces for away team wins
            fig.add_trace(go.Bar(
                x=away_wins_count_by_category.index,
                y=away_wins_count_by_category.values,
                name='Away Team Wins',
                marker_color='red',
                text=away_wins_percentage_by_category.round(2).astype(str) + '%',
                textposition='auto'
            ))

            # Update layout
            fig.update_layout(
                title='Number of Wins by Points Difference',
                xaxis_title='Points Difference',
                yaxis_title='Number of Wins',
                xaxis=dict(tickvals=home_wins_count_by_category.index, ticktext=home_wins_count_by_category.index),
                barmode='stack',
                title_x=0.5, 
                title_y=0.9, 
                title_font_size=24
            )
            
        else:
            # Get the top ten players for the chosen column
            sorted_players = players_stats_df.groupby('PLAYER_NAME').mean().sort_values(by=chosen_col, ascending=True)

            x = sorted_players.index.get_level_values('PLAYER_NAME').tolist()[-10:]
            y = sorted_players[chosen_col].tolist()[-10:]

            # Define colors of the graph
            graph_colors = [next(team_color_palette) for _ in range(10)]

            # Add trace
            fig.add_trace(go.Bar(x=x, y=y, marker_color=graph_colors))
            
            # Update layout
            fig.update_layout(title=f'Top Ten Best Payers of the League at {chosen_col}',
                            xaxis_title='Player Name',
                            yaxis_title=f'Average {chosen_col} Per Game',
                            title_x=0.5,
                            title_y=0.9,
                            title_font_size=24,
                            xaxis_tickangle=-45)
            
    else :
        if (chosen_col == 'team_stats'):
            # Filter data for the chosen team
            team_df = all_games_df.loc[all_games_df['NICKNAME'] == chosen_team]

            # Group by season and calculate win percentage
            win_loss_df = team_df.groupby('SEASON').agg({'WIN': ['sum', 'count']})
            win_loss_df.columns = ['WIN', 'TOTAL']

            # Calculate win percentage
            win_loss_df['LOSS'] = win_loss_df['TOTAL'] - win_loss_df['WIN']

            # Extract season (index) and win percentage values
            x = win_loss_df.index.tolist()

            fig.add_trace(go.Bar(x=x, y=win_loss_df['WIN'], name='Win', marker_color=team_colors[chosen_team][0]))
            fig.add_trace(go.Bar(x=x, y=win_loss_df['LOSS'], name='Loss', marker_color=team_colors[chosen_team][1]))

            # Update layout
            fig.update_layout(title='Win and Loss Through Seasons', xaxis_title='Season Year', yaxis_title='Win and Loss Count', barmode='stack',
                title_x=0.5, title_y=0.9, title_font_size=24, xaxis=dict(tickmode='array', tickvals=x, ticktext=x, tickangle=-45)
            )

        else:
            # Filter the DataFrame for the given team
            team_df = players_stats_df.loc[players_stats_df.index.get_level_values('TEAM_NAME') == chosen_team]


            # Sort the DataFrame by efficiency in descending order and get the top 10 rows
            sorted_players = team_df.sort_values(by='PTS', ascending=False)

            x = sorted_players.index.get_level_values('PLAYER_NAME').tolist()[:10]
            y = sorted_players['PTS'].tolist()[:10]
            seasons = sorted_players.index.get_level_values('SEASON').tolist()[:10]

            # Define colors of the graph
            graph_colors = [next(team_color_palette) for _ in range(10)]

            # Create trace for each season
            traces = []
            for i, season in enumerate(seasons):
                traces.append(go.Bar(x=[x[i]], y=[y[i]], name=season, text=[season], textposition='auto', marker_color=graph_colors[i], showlegend=False))

            # Add traces to the figure
            for trace in traces:
                fig.add_trace(trace)

            # Update layout
            fig.update_layout(title='Top Players at Average Points in a Season',
                            xaxis_title='Player Name',
                            yaxis_title='Points',
                            title_x=0.5,
                            title_y=0.9,
                            title_font_size=24,
                            barmode='stack')

    return fig

# Boxplot function
def generate_boxplots(home_dataframe, away_dataframe, column):
    fig = go.Figure()
    fig.add_trace(go.Box(y=home_dataframe[column], name='Home', boxmean=True))
    fig.add_trace(go.Box(y=away_dataframe[column], name='Away', boxmean=True))
    fig.update_layout(title=f'Boxplot of {column}', yaxis_title=column, xaxis_title='Location',title_x=0.5, title_y=0.9, title_font_size=24)
    return fig

# App callback for top right graphs
@app.callback(
    Output('top-right-graph', 'figure'),
    [Input('feature-dropdown', 'value'),
     Input('team-dropdown', 'value')]
)

def update_top_right(chosen_col, chosen_team):
    # Create the figure
    fig = fig = go.Figure()

    # Teams colors for graph
    team_color_palette = cycle(team_colors[chosen_team])

    if (chosen_team == 'All Teams'):
        if chosen_col == 'ARENA_CAPACITY':
            fig.add_trace(go.Box(y=home_games_df.loc[home_games_df["ARENA_CAPACITY"]>0][chosen_col], name='Home', marker_color='blue', boxmean=True))
            fig.update_layout(title=f'Boxplot of {chosen_col}', yaxis_title=chosen_col, xaxis_title='Location',title_x=0.5, title_y=0.9, title_font_size=24, showlegend=False)
        else :
            fig.add_trace(go.Box(y=home_games_df[chosen_col], name='Home', marker_color='blue', boxmean=True))
            fig.add_trace(go.Box(y=away_games_df[chosen_col], name='Away', marker_color='red', boxmean=True))
            fig.update_layout(title=f'Boxplot of {chosen_col} at Home and Away', yaxis_title=chosen_col, xaxis_title='Location',title_x=0.5, title_y=0.9, title_font_size=24, showlegend=False)

    else:
        if (chosen_col == 'team_stats'):
            # Filter data for the chosen team
            team_df = all_games_df.loc[all_games_df['NICKNAME'] == chosen_team]

            # Group by season and calculate points average
            team_df = team_df.groupby('SEASON').agg({'PTS': 'mean'})

            # Extract season (index) and points average values
            x = team_df.index.tolist()
            y = team_df['PTS'].tolist()

            # Add trace
            fig.add_trace(go.Scatter(x=x, y=y, marker_color=team_colors[chosen_team][1]))

            # Update layout
            fig.update_layout(
                title='Average Points Through Seasons',
                xaxis_title='Season Year',
                yaxis_title='Average Points',
                title_x=0.5,
                title_y=0.9,
                title_font_size=24,
                xaxis_tickangle=-45
            )
        else:
            # Define colors of the graph
            graph_colors = [next(team_color_palette) for _ in range(19)]

            # Filter data for the chosen team
            stats_df = players_stats_df.loc[players_stats_df.index.get_level_values('TEAM_NAME') == chosen_team]

            # Sort the DataFrame by the index (season) to ensure alignment
            stats_df = stats_df.sort_index(level='SEASON')

            # Get the index of the maximum FG_PCT for each season
            max_fg_pct_index = stats_df.groupby('SEASON')['FG_PCT'].idxmax()

            # Extract the player names and corresponding maximum FG_PCT for each season
            player_names_fg_pct = [idx[0] for idx in max_fg_pct_index]
            best_fg_pct = [stats_df.loc[idx, 'FG_PCT']*100 for idx in max_fg_pct_index]

            # Extract data for each stat
            x = stats_df.index.get_level_values('SEASON').unique().tolist()

            # Create bar trace for FG_PCT
            bar_trace_fg_pct = go.Bar(
                x=x,
                y=best_fg_pct,
                name='FG_PCT',
                marker=dict(color=graph_colors),
                hoverinfo='text',
                text=player_names_fg_pct,
                textposition='inside',
                textangle=90,
                textfont=dict(size=14)
            )

            fig = go.Figure(data=[bar_trace_fg_pct])

            # Update layout
            fig.update_layout(
                title='Players Best FG_PCT Average Over Seasons',
                xaxis_title='Season Year',
                yaxis_title='FG_PCT Average',
                title_x=0.5,
                title_y=0.9,
                title_font_size=24,
                xaxis=dict(tickmode='array', tickvals=x, ticktext=x),
                legend=dict(orientation='h', yanchor='bottom', y=-0.3, x=0.44),
                xaxis_tickangle=-45
            )

            
    return fig


@app.callback(
    Output('bottom-graph', 'figure'),
    [Input('feature-dropdown', 'value'),
     Input('team-dropdown', 'value')]
)

def update_bottom(chosen_col, chosen_team):
    # Create a figure
    fig = go.Figure()

    # Teams colors for graph
    team_color_palette = cycle(team_colors[chosen_team])

    if (chosen_team == 'All Teams'):

        if chosen_col == 'ARENA_CAPACITY':            
            # Sort teams based on ARENA_CAPACITY
            arena_capacity_df = home_games_df.loc[home_games_df["ARENA_CAPACITY"]>0].groupby('NICKNAME')[['ARENA_CAPACITY','WIN']].mean().reset_index()
            sorted_arenaCapacity = arena_capacity_df.sort_values(by='ARENA_CAPACITY')
            
            # Add trace
            fig.add_trace(go.Scatter(x=sorted_arenaCapacity['ARENA_CAPACITY'], y=sorted_arenaCapacity['WIN']*100, mode='markers+lines',
                                      marker_color='blue', marker_symbol='circle', line=dict(color='blue', width=1)))
            
            # Update layout
            fig.update_layout(title='Home Win Percentage Depending on Arena Capacity',
                            xaxis_title='Arena Capacity',
                            yaxis_title='Home Win Percentage (%)',
                            title_x=0.5,
                            title_y=0.9,
                            title_font_size=24)
        else:
            # Calculate average for home and away games
            home_avg = home_games_df.groupby('NICKNAME')[chosen_col].mean().reset_index()
            away_avg = away_games_df.groupby('NICKNAME')[chosen_col].mean().reset_index()
            
            # Sort teams based on the chosen column
            sorted_home_avg = home_avg.sort_values(by=chosen_col)
            # Sort df1 and get the sorted index
            sorted_home_index = sorted_home_avg.index

            # Use the sorted index to sort df2
            sorted_away_avg = away_avg.loc[sorted_home_index]
            
            # Add traces for home games
            fig.add_trace(go.Bar(x=sorted_home_avg['NICKNAME'], y=sorted_home_avg[chosen_col], name='Home', marker_color='blue', legendgroup='Home', offsetgroup=0))
            
            # Add traces for away games (for other variables)
            fig.add_trace(go.Bar(x=sorted_home_avg['NICKNAME'], y=sorted_away_avg[chosen_col], name='Away', marker_color='red', legendgroup='Away', offsetgroup=1))
            
            # Update layout
            fig.update_layout(title=f'Comparison of Average {chosen_col} in Home and Away Games',
                            xaxis_title='Team',
                            yaxis_title=f'Average {chosen_col}',
                            barmode='group',
                            legend_title='Game Location',
                            legend=dict(orientation='h', yanchor='bottom', y=-0.5, xanchor='left'),
                            title_x=0.5,
                            title_y=0.9,
                            title_font_size=24,
                            xaxis_tickangle=-45)
            
    else:
        if (chosen_col == 'team_stats'):
            # Define colors of the graph
            graph_colors = [next(team_color_palette) for _ in range(19)]

            # Filter data for the chosen team
            team_players_df = cumul_efficiency_df.loc[(cumul_efficiency_df.index.get_level_values('TEAM_NAME') == chosen_team)]

            # Group by season and sum
            team_players_df = team_players_df.groupby('SEASON').sum()

            x = team_players_df.index.get_level_values('SEASON').tolist()
            y = team_players_df['EFFICIENCY'].tolist()

            # Add trace
            fig.add_trace(go.Bar(x=x, y=y, marker_color=graph_colors))
            
            # Update layout
            fig.update_layout(title='Team Cumulative Efficiency Average By Game Through Seasons', xaxis_title='Season Year', yaxis_title='Cumulative Efficiency Average',
                            title_x=0.5, title_y=0.9, title_font_size=24, xaxis=dict(tickmode='array', tickvals=x, ticktext=x))
            
        else:
            # Filter data for the chosen team
            stats_df = players_stats_df.loc[players_stats_df.index.get_level_values('TEAM_NAME') == chosen_team]

            # Sort the DataFrame by the index (season) to ensure alignment
            stats_df = stats_df.sort_index(level='SEASON')

            # Get the index of the maximum value for each season in each statistic column
            max_pts_index = stats_df.groupby('SEASON')['PTS'].idxmax()
            max_ast_index = stats_df.groupby('SEASON')['AST'].idxmax()
            max_reb_index = stats_df.groupby('SEASON')['REB'].idxmax()

            # Extract the player names and corresponding maximum values for each season
            player_names_pts = [idx[0] for idx in max_pts_index]
            best_pts = [stats_df.loc[idx, 'PTS'] for idx in max_pts_index]

            player_names_ast = [idx[0] for idx in max_ast_index]
            best_ast = [stats_df.loc[idx, 'AST'] for idx in max_ast_index]

            player_names_reb = [idx[0] for idx in max_reb_index]
            best_reb = [stats_df.loc[idx, 'REB'] for idx in max_reb_index]

            # Extract data for each stat
            x = stats_df.index.get_level_values('SEASON').unique().tolist()

            # Define stat labels
            stats_labels = ['PTS', 'AST', 'REB']

            team_color_palette = [team_colors[chosen_team][0], team_colors[chosen_team][1], 'grey']

            # Create grouped bar traces with specified colors
            bar_traces = []

            for label, color in zip(stats_labels, team_color_palette):
                if label == 'PTS':
                    y_values = best_pts
                    text_values = player_names_pts
                elif label == 'AST':
                    y_values = best_ast
                    text_values = player_names_ast
                else:
                    y_values = best_reb
                    text_values = player_names_reb

                # Create bar trace
                bar_trace = go.Bar(
                    x=x,
                    y=y_values,
                    name=label,
                    marker=dict(color=color),
                    hoverinfo='text',
                    text=text_values,
                    textposition='inside',
                    textangle=90,
                    textfont=dict(size=14)

                )
                bar_traces.append(bar_trace)

            fig = go.Figure(data=bar_traces)

            # Update layout
            fig.update_layout(
                title='Players Best Stats Over Seasons',
                xaxis_title='Season Year',
                yaxis_title='Best Stats of the Year',
                barmode='group',
                title_x=0.5,
                title_y=0.9,
                title_font_size=24,
                xaxis=dict(tickmode='array', tickvals=x, ticktext=x),
                legend=dict(orientation='h', yanchor='bottom', y=-0.3, x=0.44),
            )

    return fig


######################################################################################################################################################################
                                                             # Prediction Models #
######################################################################################################################################################################


# Calculate prediction results for each season
all_pred_results = dash_prediction.predict_allWins()

# Define callback to update the prediction title
@app.callback(
    Output('prediction-title', 'children'),
    Input('prediction-dropdown', 'value')
)
def update_title(rank_type):
    if (rank_type == 'wins'):
        # Update the title text based on the rank type
        title_text = 'Win Prediction'
    
    else:
        # Update the title text based on the rank type
        title_text = 'Rank Prediction' 

    return html.Span(title_text)

# Define a callback to update the prediction results table for the West conference
@app.callback(
    Output('prediction-left-graph', 'figure'),
    [Input('prediction-year-dropdown', 'value'),
    Input('prediction-dropdown', 'value')]
)
def update_prediction_left(chosen_season, rank_type):
    # Calculate the index in the all results dataframe corresponding to the chosen season
    index = int(chosen_season) - 2006
    if index > 5:
        index -= 2
    # Get the results prediction dataframe
    pred_results_df = all_pred_results[index]

    # Filter DataFrame for West conference
    west_results = pred_results_df[pred_results_df['Conference'] == 'West']

    # Sort the West DataFrame by predicted wins and actual wins to get the rankings
    west_results_sorted = west_results.sort_values(by=['Actual Wins', 'Predicted Wins'], ascending=False)

    # Drop conference column
    west_results_sorted = west_results_sorted.drop('Conference', axis=1)

    if(rank_type == 'wins'):
        # Create a Plotly table from the DataFrame
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(west_results_sorted.columns),
                        fill_color='#F0F0F0',
                        font=dict(color='black', size=20),  # Larger font for column names
                        align='center',
                        height=40),
            cells=dict(values=[west_results_sorted[col] for col in west_results_sorted.columns],
                    fill_color=[[team_colors[row][0] for row in west_results_sorted['Team Name']]],
                    font=dict(color='black', size=16),  # Larger font for table cells
                    align='center',
                    height=30))
        ])

        # Update layout
        fig.update_layout(
            title=f"<b style='font-size: 30px; color: #CE1141'>Western Conference Win Prediction Results</b><br>Season {chosen_season} / {str(int(chosen_season) + 1)}",
            title_x=0.5, 
            margin=dict(t=100, l=0, r=0, b=0), 
            height=600,  # Increase table height
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
        )

    else:
        # Add columns for predicted wins ranking and actual wins ranking with integer values
        west_results_sorted['Actual Rank'] = west_results_sorted['Actual Wins'].rank(ascending=False, method='min').astype(int)
        west_results_sorted['Predicted Rank'] = west_results_sorted['Predicted Wins'].rank(ascending=False, method='min').astype(int)

        # Drop wins predictions columns
        west_results_sorted = west_results_sorted[['Team Name', 'Actual Rank', 'Predicted Rank']]

        # Create a Plotly table from the DataFrame
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(west_results_sorted.columns),
                        fill_color='#F0F0F0',
                        font=dict(color='black', size=20),  # Larger font for column names
                        align='center',
                        height=40),
            cells=dict(values=[west_results_sorted[col] for col in west_results_sorted.columns],
                    fill_color=[[team_colors[row][0] for row in west_results_sorted['Team Name']]],
                    font=dict(color='black', size=16),  # Larger font for table cells
                    align='center',
                    height=30))
        ])

        # Update layout
        fig.update_layout(
            title=f"<b style='font-size: 30px; color: #CE1141'>Western Conference Ranking Prediction Results</b><br>Season {chosen_season} / {str(int(chosen_season) + 1)}",
            title_x=0.5, 
            margin=dict(t=100, l=0, r=0, b=0), 
            height=600,  # Increase table height
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
        )

    return fig

# Define a callback to update the prediction results table for the East conference
@app.callback(
    Output('prediction-right-graph', 'figure'),
    [Input('prediction-year-dropdown', 'value'),
    Input('prediction-dropdown', 'value')]
)

def update_prediction_right(chosen_season, rank_type):
    # Calculate the index in the all results dataframe corresponding to the chosen season
    index = int(chosen_season) - 2006
    if index > 5:
        index -= 2
    # Get the results prediction dataframe
    pred_results_df = all_pred_results[index]

    # Filter DataFrame for East conference
    east_results = pred_results_df[pred_results_df['Conference'] == 'East']

    # Sort the East DataFrame by predicted wins and actual wins
    east_results_sorted = east_results.sort_values(by=['Actual Wins', 'Predicted Wins'], ascending=False)

    # Drop conference column
    east_results_sorted = east_results_sorted.drop('Conference', axis=1)

    if(rank_type == 'wins'):
        # Create a Plotly table from the DataFrame
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(east_results_sorted.columns),
                        fill_color='#F0F0F0',
                        font=dict(color='black', size=20),  # Larger font for column names
                        align='center',
                        height=40),
            cells=dict(values=[east_results_sorted[col] for col in east_results_sorted.columns],
                fill_color=[[team_colors[row][0] for row in east_results_sorted['Team Name']]],
                font=dict(color='black', size=16),  # Larger font for table cells
                align='center',
                height=30))
        ])

        # Update layout
        fig.update_layout(
            title=f"<b style='font-size: 30px; color: #00538C;'>Eastern Conference Win Prediction Results</b><br>Season {chosen_season} / {str(int(chosen_season) + 1)}",
            title_x=0.5,
            margin=dict(t=100, l=0, r=0, b=0),  # Adjust margins
            height=600,  # Increase table height
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
        )

    else:
        # Add columns for predicted wins ranking and actual wins ranking with integer values
        east_results_sorted['Actual Rank'] = east_results_sorted['Actual Wins'].rank(ascending=False, method='min').astype(int)
        east_results_sorted['Predicted Rank'] = east_results_sorted['Predicted Wins'].rank(ascending=False, method='min').astype(int)

        # Drop wins predictions columns
        east_results_sorted = east_results_sorted[['Team Name', 'Actual Rank', 'Predicted Rank']]

        # Create a Plotly table from the DataFrame
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(east_results_sorted.columns),
                        fill_color='#F0F0F0',
                        font=dict(color='black', size=20),  # Larger font for column names
                        align='center',
                        height=40),
            cells=dict(values=[east_results_sorted[col] for col in east_results_sorted.columns],
                    fill_color=[[team_colors[row][0] for row in east_results_sorted['Team Name']]],
                    font=dict(color='black', size=16),  # Larger font for table cells
                    align='center',
                    height=30))
        ])

        # Update layout
        fig.update_layout(
            title=f"<b style='font-size: 30px; color: #00538C'>Eastern Conference Ranking Prediction Results</b><br>Season {chosen_season} / {str(int(chosen_season) + 1)}",
            title_x=0.5, 
            margin=dict(t=100, l=0, r=0, b=0), 
            height=600,  # Increase table height
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
        )

    return fig


# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='localhost')

