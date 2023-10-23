from utlis import *  # import packages and functions for preprocess
data = get_data( "train" ) # fetch training dataset

for target in [ "content", "wording" ]:
    params = config['best_params'][target] # Parameters selected by GridSearch with cross-validation
    model = XGBRegressor( **params )

    model.fit( data[config['best_features_arrange'][target]], data[target] ) # fit model on whole dataset

    model.save_model( f"best_model_fitted_whole_{target}.json" ) # store trained model on drive