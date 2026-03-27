import pandas as pd 
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split 		
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder




data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')


cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)




numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[('num', numerical_transformer, numerical_cols),('cat', categorical_transformer, categorical_cols)])



my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=1)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)


my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', my_model)
                             ])

my_pipeline.fit(X_train, y_train)			# Preprocessing of training data, fit model 
preds = my_pipeline.predict(X_valid)			# Preprocessing of validation data, get predictions
score = mean_absolute_error(y_valid, preds)		# Evaluate the model



def get_score():    #cross validation
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()




predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(y_valid, predictions)))
