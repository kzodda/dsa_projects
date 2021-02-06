'''
Karl Zodda

The goal of this Jupyter Notebook is to look at that data and understand it. Then move on to using seaborn and pandas methods to gather information on distribution of what hand players shoot with. Then the data is broken down to the years 1964-1974 (The UCLA Golden Era a.k.a the height of the John Wooden Era) and looking at if the draft picks from the national championship schools and when they were taken in the draft after completing their national title winning season. Then after breaking the data into easy to use data for modeling, we try to predict the PER(player efficiency rating) of a player based on their other stats. This is a Regression Task. 

PER is an all-in-one basketball rating created by John Hollingers which takes into account all of a player's contributions. It strives to measure a player's per-minute performance adjusted for pace. This is to be able to indicate a player's value regardless of position. This the holy grail performance metric. We want to be able to predict it because then organizations can allocate more resources towards players with higher predicted PER's. 
'''
## Importing packages


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import utils
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

'''
## Opeining up the file. 
Nba Salaries and players
Creating DataFrames
'''
## Reading in the file as a pandas DataFrame
player_info = pd.read_csv(r'players-nba-salaries-QueryResult.csv')

# Taking an overall look at the data to get an idea of what we are looking at
player_info.head(6)

player_info.info()
# Now we are seeing different datatypes.
# We have 1 integers,
#9 floats,
#& 14 objects (Will Have to mess with the objects)

player_info.shape
## We have 4685 entries. Quite a bit of data

#Looking to see if we have any null values
player_info.isnull().sum()
'''
We will have to take care of the null values
to build a better model
'''

## Describing the data so we can the distribution of numerical data
player_info.describe()
## This is just a description of the data.

##Looking at the correlation between numerical features
player_info.corr()
# We even have a new negative correlations

## A list of our columns just for reference
player_info.columns

# A description of our columns in the same order as columns above
## ID, Birthdate, Career assists, Career fieldgoals, Career 3's, Career free throw percentage, Career games, Career Player efficiency rating, Career Points, Career Total Rebounds, Career Wins, Career effective field goal percentage, College, Draft Pick, Draft Rounds, Drafted by, Draft Year, height, Highschool, Name, Position, Shoots, Weight. 


## Looking at Career_pts Averaged by Game over their careers and their shot hand

## A violin plot to show the distribution in an easy to understand format
sns.violinplot(x='shoots', y='career_pts', data=player_info, dodge=True)

## Looking at the distributions, there are a lot of right-handed players at the bottom with a very few at the top. There is a sudden decrease in the amount of players as career points goes up. Left-handed shooters have a more gradual decline of number of players as career points goes up. The best shooter is a right handed shooter.

## Looking at the Draft Round and Draft Position of players from the Championship schools during the John Wooden Era

## Getting a df of just the years 1964-1974 (John Wooden Era)
ucla_era = player_info[player_info.draft_year.isin(['1964', '1965', '1966', '1967', '1968', '1969', '1970',
                                                    '1971', '1972', '1973', '1974', '1975'])].copy()
ucla_era.shape
## Taking a look at the shape so we know how much data we have

## Making the Data More Manageable.

## Changing the draft round object data to integer data


def extract_round(str_val):
    return int("".join([n for n in str_val if n.isnumeric()]))


# Apply the function to the draft_round column
ucla_era['draft_round'] = ucla_era['draft_round'].apply(extract_round)

# Take quick look at results
ucla_era.sample(5)
## We want it to be an integer for our numercial plots

'''
A list of the NCAA Championship Schools during the UCLA ERA:

1964- Univeristy of California, Los Angeles
1965- Univeristy of California, Los Angeles
1966- University of Texas at El Paso
1967- Univeristy of California, Los Angeles
1968- Univeristy of California, Los Angeles
1969- Univeristy of California, Los Angeles
1970- Univeristy of California, Los Angeles
1971- Univeristy of California, Los Angeles
1972- Univeristy of California, Los Angeles
1973-Univeristy of California, Los Angeles
1974- North Carolina State Univeristy
1975- Univeristy of California, Los Angeles
'''

## A df of players that got drafted after a title winning season between 1964-1974
ucla_era_champschools = ucla_era[((ucla_era.draft_year.isin(['1964', '1965', '1967',
                                                             '1968', '1969', '1970',
                                                             '1971', '1972', '1973', '1975'])) &
                                  (ucla_era.college == 'University of California, Los Angeles'))
                                 | ((ucla_era.college == 'University of Texas at El Paso') &
                                    (ucla_era.draft_year == '1966'))
                                 | ((ucla_era.college == 'North Carolina State Univeristy') &
                                    (ucla_era.draft_year == '1974'))].copy()
ucla_era_champschools.head()
'''
 We want a Dataframe of just these players to determine how that played an impact
on their draft round selection.
'''

## A df of players that got drafted after not having a title winning season 1964-1974
ucla_era_non_champschools = ucla_era[((ucla_era.draft_year.isin(['1964', '1965', '1967', '1968', '1969', '1970',
                                                                 '1971', '1972', '1973', '1975'])) &
                                      (ucla_era.college != 'University of California, Los Angeles'))
                                     | ((ucla_era.college != 'University of Texas at El Paso') &
                                        (ucla_era.draft_year == '1966'))
                                     | ((ucla_era.college != 'North Carolina State Univeristy') &
                                        (ucla_era.draft_year == '1974'))].copy()
ucla_era_non_champschools.head()
'''
Here is a DataFrame of players who did not win a national championship
So we can compare it to the players who did win one.
'''

### LOOK AT CHAMP SCHOOL PLAYERS AND THEIR DRAFT ROUND and Draft Team
plt.figure(figsize=(5,20))
sns.stripplot(x = 'draft_round', 
              y = 'draft_team', data = ucla_era_champschools, 
              hue = 'draft_year');
'''
We can see what round players were taken and by what team to get
a clear picture
'''

plt.figure(figsize=(10, 40))
sns.stripplot(x='draft_round',
              y='draft_team', data=ucla_era_non_champschools,
              hue='draft_year')
'''
Here we have a graph showing what round players were taken 
and by what team. These are players that did not win national
championships their draft year.
'''


# Here is a distribution plot of championship winning players getting drafted
sns.distplot(ucla_era_champschools.draft_round, kde = False, bins = 4);

# Here is a distribution plot of non-championship winning players getting drafted
sns.distplot(ucla_era_non_champschools.draft_round, kde=False, bins=7);

'''
With all 14 players who were drafted coming off a championship year coming from UCLA, We can also see the largest proportion were drafted in the first round. 
'''

## Looking at the first round draft picks coming from champ schools
ucla_era_firstround_champions = ucla_era_champschools[ucla_era_champschools.draft_round == 1].copy(
)
ucla_era_firstround_champions.head(7)
## We want to look at just the first round picks to see where they were drafted.
## Only looking at the championship winning players to see how that boosted their draft stock.
## Or to see if they won due to superstars.

## Applying a function to convert draft pick # to an integer instead of an object
## Doing this so we can use numerial plots instead of categorical plots.


def overall_converter(subset):
    subnum = ''
    for n in subset:
          if n in '1234567890':
            subnum += n
    return int(subnum)


ucla_era_firstround_champions['overall_pick'] = ucla_era_firstround_champions.draft_pick.apply(overall_converter)
ucla_era_firstround_champions.drop(['draft_pick'], axis=1, inplace = True)
ucla_era_firstround_champions.rename(columns = {'overall_pick': 'draft_pick'}, inplace = True)
ucla_era_firstround_champions.head()

## A strip plot showing where the different players were taken
## Different colours to show different draft years. 
sns.stripplot(x = 'draft_round',
              y= 'draft_pick', data = ucla_era_firstround_champions,
              hue = 'draft_year');

'''
You can see that after winning the 1969 National Championship two players were taken early in the draft. Those players happen to be Kareem Abdul-Jabbar and Allen Lucius
'''              

'''
# Model Building
Now we are going to build a model that predict the PER of players from the John Wooden Era. 
'''
##Creating the df of the section we are going to use for our model
ucla_era_formodel = player_info[player_info.draft_year.isin(['1964', '1965','1966','1967',
                                                             '1968','1969', '1970',
                                                    '1971', '1972','1973', 
                                                             '1974', '1975' ])].copy()
print(ucla_era_formodel.shape)
'''
We are still going to look at players from our area of interest which
is the height of the John Wooden Era (1964-1975)
'''


## Converting height Object to Integer
## We want it to be an integer so we don't use one_hot encoding/ binning. 
def height_converter (subset):
    return (int(subset[0])* 12) + int(subset[2])
    
ucla_era_formodel['height_inches'] = ucla_era_formodel.height.apply(height_converter)ucla_era_formodel.drop(['height'], axis = 1, inplace = True)

ucla_era_formodel.head()


## Dropping Unnecessary Columns that play no role in our model
ucla_era_formodel.drop(['birthdate'], axis = 1, inplace = True)
ucla_era_formodel.drop(['birthplace'], axis = 1, inplace = True)
ucla_era_formodel.drop(['draft_year'], axis = 1, inplace = True)
ucla_era_formodel.drop(['highschool'], axis = 1, inplace = True)
ucla_era_formodel.drop(['id'], axis = 1, inplace = True)
ucla_era_formodel.drop(['name'], axis = 1, inplace = True)
ucla_era_formodel.drop(['college'], axis = 1, inplace = True)
ucla_era_formodel.drop(['draft_pick'], axis = 1, inplace = True)


## Coverting Weight Object to Integer to make it easier for model prediciton
def convert_weight (subset):
    return int(subset[:3])
## Applying the Function and naming it denoting the measurements
ucla_era_formodel['weight_lbs'] = ucla_era_formodel.weight.apply(convert_weight)
## Dropping the original Weight column
ucla_era_formodel.drop(['weight'], axis = 1, inplace = True)

ucla_era_formodel.head()

## Checking to see the different datatypes.
## We want some of the columns to remain as objects
ucla_era_formodel.info()


'''
## Now we are going to change the null values to Mean Values to not obstruct our model building and without resorting to dropping the row.
'''

## Checking which values are null
ucla_era_formodel.isnull().sum()


## Finding the mean of each of the columns that has null values
career_fg3_mean = ucla_era_formodel.career_fg3.mean()
career_ft_mean = ucla_era_formodel.career_ft.mean()
career_efg_mean = ucla_era_formodel.career_efg.mean()
career_fg_mean = ucla_era_formodel.career_fg.mean()
## Applying the mean # to each null value
ucla_era_formodel.career_fg3.fillna(career_fg3_mean, inplace = True)
ucla_era_formodel.career_ft.fillna(career_ft_mean, inplace = True)
ucla_era_formodel.career_efg.fillna(career_efg_mean, inplace = True)
ucla_era_formodel.career_fg.fillna(career_fg_mean, inplace = True)
##Dropping the row where the shooter's hand in Nan
## It won't affect our data tremendously dropping one value
ucla_era_formodel.dropna(axis = 0, inplace = True)


## One-hot Encoding the Object values. Including: position, draft_team, draft_round, and shot

## One-hot encoding all the oject columns. 
processed_data = pd.get_dummies(ucla_era_formodel)
processed_data.head()

## MODEL BUILDING

##Creating our Target which is PER
processed_target = processed_data.career_per
## Dropping our target value from our data that will build the model
processed_data.drop(['career_per'], axis =1, inplace = True)

# Can use .to_numpy()
## Setting our X, y variables for our train test split
X = processed_data.values
y = processed_target.values.ravel()

#First Train Test Split 
'''
Didn't feel the need to stratify the data. 
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

## Scaling our data so we don't have crazy outliers. 
## Using MinMaxScaler because it is the best for our data 
scaler = MinMaxScaler()
scaler.fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)

## Threefold Split 
# We need another split of data for evaluating the models

X_trained, X_valid, y_trained, y_valid = train_test_split(train_scaled, y_train)

## The Models used are Lasso, Random Forest Regressor, and GradientBoosting Regressor

## Lasso
lasso_alpha = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 10000])
max_iter = 100000
best_score = 0.00001
average_parameters= {}
average_of_all = []
for alpha in lasso_alpha:
    lasso = Lasso(alpha = alpha, max_iter = max_iter)
    scores = cross_val_score(lasso, X_trained, y_trained, cv = 5)
    score = np.mean(scores)
    average_parameters[alpha] = score
    average_of_all.append(score)
    if score > best_score:
        best_score = score
        best_parameters = {'alpha': alpha}

print('Average of all parameters: ', np.mean(average_of_all))
print('Best Parameter and Score for that Parameter: ', best_parameters, best_score)
# print('Score of each parameter: ', average_parameters)
bestlasso = Lasso(**best_parameters, max_iter = max_iter)
bestlasso.fit(X_train, y_train)
lasso_scores = cross_val_score(bestlasso, X_test, y_test, cv = 10)
lasso_avg = np.mean(lasso_scores)
print('Best Parameters and Average Score for that Parameter: ', best_parameters, lasso_avg)
print(bestlasso)

## Random Forest Regressor
rf_n_estimators = np.array(np.linspace(1, 10, 40))
rf_new = []
for _ in rf_n_estimators:
    rf_new.append(int(_))
rf_arr = np.array(rf_new)
rf_max_depth = np.array(range(1,5))
best_score = 0.00001
parameter_score = {}
average_of_all = []
for n_estimators in rf_arr:
    for mx_depth in rf_max_depth:
        rfr = RandomForestRegressor(n_estimators = n_estimators, max_depth = mx_depth)
        scores = cross_val_score(rfr, X_trained, y_trained, cv = 5)
        score = np.mean(scores)
        parameter_score[n_estimators, mx_depth] = score
        average_of_all.append(score)
        if score > best_score:
            best_score = score
            best_parameters = {'n_estimators': n_estimators, 
                              'max_depth' : mx_depth}

print('Average of all Parameters: ', np.mean(average_of_all))
print('Best Parameter & Score for that Parameter: ', best_parameters, best_score)
#print('Score of each combination (n_estimators, mx_depth): ', parameter_score)
bestrfr = RandomForestRegressor(**best_parameters)
bestrfr.fit(X_train, y_train)
bestrfr_scores = cross_val_score(bestrfr, X_test, y_test, cv = 10)
rfr_avg = np.mean(bestrfr_scores)
print('Best Parameters and Average Score for that Parameter: ', best_parameters, rfr_avg)
print(bestrfr)

## Gradient Boosting Regressor
learning_rate = np.array(np.linspace(1, 100, 10))# The higher the more corrections that can be made
n_estimators = np.array(range(1,30)) # The amount of trees 
max_depth = rf_max_depth
lr_new = []
for _ in learning_rate:
    lr_new.append(int(_))
lr_arr = np.array(lr_new)
best_score = 0.00001
parameter_score = {}
average_of_all = []
for rate in lr_new:
    for n_estimator in n_estimators:
        for mx_depth in max_depth:
            gbr = GradientBoostingRegressor(n_estimators = n_estimator, max_depth = mx_depth, learning_rate = rate)
            scores = cross_val_score(gbr, X_trained, y_trained, cv = 5)
            score = np.mean(scores)
            parameter_score[rate, n_estimator, mx_depth] = score
            average_of_all.append(score)
            if score > best_score:
                best_score = score
                best_parameters = {'learning_rate': rate,
                                   'n_estimators': n_estimator, 
                                   'max_depth' : mx_depth}

print('Average of all Parameters: ', np.mean(average_of_all))
print('Best parameter & Score for that Parameter: ', best_parameters, best_score)
#print('Score of each combination (rate, n_estimators, mx_depth): ', parameter_score)
bestgbr = GradientBoostingRegressor(**best_parameters)
bestgbr.fit(X_train, y_train)
gbrscores = cross_val_score(bestgbr, X_test, y_test, cv = 10)
gbr_avg = np.mean(gbrscores)
print('Best Parameters and Average Score for that Parameter: ', best_parameters, gbr_avg)
print(bestgbr)


## After looking at our model building, A Random Forest model is the best model to work off of. More tuning is needed in order to increase the scores. 
