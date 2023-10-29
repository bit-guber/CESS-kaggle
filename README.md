# CommonLit - Evaluate Student Summaries check [on](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries)

### this solution got me `top 28 place` on Efficient LeaderBoard [here](https://www.kaggle.com/code/ryanholbrook/evaluate-student-summaries-efficiency-lb) as name as `Bit_Guber`

## Goal of the Competition

The goal of this competition is to assess the quality of summaries written by students in grades 3-12. You'll build a model that evaluates how well a student represents the main idea and details of a source text, as well as the clarity, precision, and fluency of the language used in the summary. You'll have access to a collection of real student summaries to train your model.

Your work will assist teachers in evaluating the quality of student work and also help learning platforms provide immediate feedback to students.

## My Solution

This Repo contains dataset and code that used to trained and inference section of competition.

- best*model_fitted_whole*[ content, wording ].json ( trained model on specific targets )
- config.json ( configuration file which contains best parameters )
- utils.py ( which contains preprocess steps )
- setup.py ( download require packages for solution )
- train.py ( optimal training python code )
- inference.py ( predict test summaries file )
- requirements.txt ( necessary packages for solution )

<br>
there few NLP Feature engineering and Gradient Boosting model that help my solution.
