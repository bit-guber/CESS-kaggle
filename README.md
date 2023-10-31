# CommonLit - Evaluate Student Summaries check [on](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries)

### this solution got me passed `RMSE 0.539` (lower is better) and `top 28 place` on Efficient LeaderBoard [here](https://www.kaggle.com/code/ryanholbrook/evaluate-student-summaries-efficiency-lb) as name as `Bit_Guber`
## Problem Overview
Summary writing is an important skill for learners of all ages. Summarization enhances reading comprehension, particularly among second language learners and students with learning disabilities. Summary writing also promotes critical thinking, and it’s one of the most effective ways to improve writing abilities. However, students rarely have enough opportunities to practice this skill, as evaluating and providing feedback on summaries can be a time-intensive process for teachers. Innovative technology like large language models (LLMs) could help change this, as teachers could employ these solutions to assess summaries quickly.

There have been advancements in the automated evaluation of student writing, including automated scoring for argumentative or narrative writing. However, these existing techniques don't translate well to summary writing. Evaluating summaries introduces an added layer of complexity, where models must consider both the student writing and a single, longer source text. Although there are a handful of current techniques for summary evaluation, these models have often focused on assessing automatically-generated summaries rather than real student writing, as there has historically been a lack of these types of datasets.

Competition host CommonLit is a nonprofit education technology organization. CommonLit is dedicated to ensuring that all students, especially students in Title I schools, graduate with the reading, writing, communication, and problem-solving skills they need to be successful in college and beyond. The Learning Agency Lab, Vanderbilt University, and Georgia State University join CommonLit in this mission.

As a result of your help to develop summary scoring algorithms, teachers and students alike will gain a valuable tool that promotes this fundamental skill. Students will have more opportunities to practice summarization, while simultaneously improving their reading comprehension, critical thinking, and writing abilities. 

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
