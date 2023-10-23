from utlis import * # import packages and functions for preprocess

test_data = get_data( "test" ) # fetch training dataset

submission = test_data.loc[ :, [ "student_id" ] ] 
initial_model = XGBRegressor() # create (dump) model
for target in [ "content", "wording" ]:
    initial_model.load_model( f"best_model_fitted_whole_{target}.json" ) # load trained model 
    submission[ target ] = initial_model.predict(  test_data[ config["best_features_arrange"][target]  ]  ) # predict specific target on test dataset

submission = submission[ [ "student_id", "content", "wording" ] ] # arrange submission format 
submission.to_csv( "submission.csv", index = False )