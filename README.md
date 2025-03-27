# Predicting Amazing Race Outcomes
![amazing_race_logo_world](https://github.com/user-attachments/assets/08b3a3be-1e85-4117-b6e7-64d087cf554c)


I love the show, [_The Amazing Race_](https://www.cbs.com/shows/amazing_race/). Having an adventurous spirit, loving to travel, and try new things, this show hits all the marks for me. And so I’ve often wondered how I would perform if I went on the show, and who should I enlist to be my partner to make the strongest team. This inspired me to actually gather some data to see if I could understand what makes a strong team and build a tool that would do just that – predict a team’s chance of winning the race.

This project is the work behind the tool that I built -- a deployed render app that you can try here: [Amazing Race Prediction App](https://amazing-race-pred-app.onrender.com/)

## Background
In case you’re unfamiliar with the show, here’s a little background. The [Amazing Race](https://en.wikipedia.org/wiki/The_Amazing_Race_(American_TV_series)) is a reality competition show in which thirteen teams of two people race around the world. Each season is split into legs, with each leg requiring teams to deduce clues, navigate themselves in foreign areas, interact with locals, perform physical and mental challenges, and travel by airplane, boat, taxi, and other public transportation options on a limited budget provided by the show.

## Data
I wanted to focus on the people in the teams as variables to success, rather than the obstacles that they faced during the race. So I knew that there were several data points that I wanted to include and investigate. The data that I wanted, included:
* The teams’ demographic makeup (age, sex, race, etc)
* Home state
* The team’s relationship to each other (siblings, romantic, parent/child, etc)
* What each person in the team does for a living
* Any information about the personality/characteristics of each member of the team
* What place did the team finish in the race
Some of these data points were readily available on wikipedia. For the others, I found a [fandom website](https://amazingrace.fandom.com/wiki/Official_Winners) for the show and scraped the data that was of interest to me.

I ended up joining the two datasets based on season and team placement. And doing a lot of cleaning for consistent formatting, to make sure the demographic categories weren’t repetitive, text data for occupation was cleaned and broken out to reflect each separate contestant on the team, and any other ‘gotchas’ in the data, such as teams that raced in multiple seasons.

* There were 414 teams from Wikipedia
* There were 350 teams scraped from fandom site
* After merging and cleaning (duplicate teams or stranger teams or family teams of more than 2 people), we’re left with 313 teams
* The data has 31 winning teams and 282 losing teams

## Analysis

### Where They’re From
* 75% of teams have contestants that come from the same state
* California teams make up a third of these ‘same state’ teams
* Top 4 states for teams from same state make up 56% of total teams (CA, NY, FL, TX)
* Wisconsin is notable in that it is one of the only low volume states that has had more than 1 winning team
  
### Age Ain’t Nothin’ But a Number (or is it?)
* The average age of losing teams is 34. Winning is 29
* The average age difference among losing teams is 6 years. Winning is 3
* In 23% of the teams, the contestants have the same occupation (71 teams). 29% of winning teams had same occupation

<img width="631" alt="amazing_race_age_distro" src="https://github.com/user-attachments/assets/bc782ff6-93bb-49ca-9276-c5ca8fd96597" />






### Other Team Stats
Weighted for volume of that particular kind of team, it’s more likely winning teams are:
* Co-ed
* Are or have been a romantic couple
* All Male

<img width="704" alt="amazing_race_team_stats" src="https://github.com/user-attachments/assets/9c58e0a1-50c7-4f84-9b29-d6d03f3bb53a" />







## Building a Basic Model
When I talk about building a basic model here, I mean that I am going to refrain from engineering any features with our text data. Some things to think about while trying to build a predictive classifier for this problem are:
* The data are small
* The target variable is imbalanced…only 31 true out of 313 total
* We have some categorical variables, such as state, which if used could greatly increase dimensionality of the data

To deal with the imbalanced data, I used three methods of resampling to compare model training results: random undersampling, random oversampling, and SMOTE. I also used kfold cross validation to try to mitigate bias in the small training set. However, I saw that with kfold, testing on our hold-out data performed leaps and bounds better than on our regular test set. This indicates that we need to be vigilant about building models on data samples where the underlying distribution does not reflect the unmanipulated data. Care must also be taken to prevent data leakage with trying to balance data and perform cross-validation. We must make sure that the sampling happens for each individual split, rather than on the data before the splits happen. Here are the model performance results:

<img width="630" alt="amazing_race_model_results-1" src="https://github.com/user-attachments/assets/0dff35d9-2f99-4374-adea-780768aa28e1" />

Because the data are imbalanced, I didn’t bother looking at accuracy as a performance measure. I focused on recall which tells us out of all the teams that actually won, what percent of those teams did the model predict as winners. We see that on average the oversampled Random Forest model performed the best, with a recall of .61 and precision of .63.

What we see in the results is that the resampling method plays a key part in the model training. It’s also interesting to note that the models trained on the imbalanced data have recall and precision in line with the balanced models, but the f1 score is of course lower. It’s important to understand what the sampling methods are doing as well. Oversampling methods, for example, can create duplicate data that make the model appear stronger in cross-validation (repeated patterns), but not perform as well on unseen data.

Possible Improvements
Ensembling
Threshold cutoffs
Feature Engineering

I also wanted to see if ensembling the models would yield better results, where some models may do better on certain cases than others. To do this, I averaged the predicted probabilities of several models togethers, and checked to see if that avg was below or above .5 to make a final prediction of Win or not win. This method did not yield much improved results. 

More investigation into probability thresholds could be another avenue of improving model performance.

The next thing that I wanted to investigate was using NLP to layer in some of our text data, such as occupations, as features into the model.


## NLP - Feature Engineering with Word2Vec
I wanted to use occupations in a model because I felt that they could be a proxy for some underlying skills or characteristics of an individual. For example, an engineer may be good at solving problems and a dancer may be agile/athletic. I felt that if I could transform these occupation strings into something that held some intrinsic meaning behind what the job was, this could possibly be a great feature in the classification model.

I immediately thought of Word2Vec. It is a model that converts tokens (words) into embeddings (vectors) by taking the weights from a continuous bag of words (CBOW) or skipgram model. A CBOW model aims to predict a word in the midst of a window of other words. Skipgram aims to predict context from a single word. At first I thought I would train my own model, but I quickly realized that I didn’t have enough data or context around the words that I had to make any meaningful embeddings. I decided it could be interesting to train a model using the text from the US Occupational Outlook Handbook. After downloading the data from the Bureau of Labor Statistics, cleaning and formatting the data, I was able to create embeddings to represent the occupations.

Plotting a sample of my occupation word vectors into 2D space using t-SNE, this is what I got. You can see that words like doctor, pharmacist, and orthopedic cluster together, while words like dispatcher, trucker, motorcyclist are close together. This gave me some confidence that the word embeddings had picked up some semantic meaning.

<img width="626" alt="amazing_race_embedding_viz" src="https://github.com/user-attachments/assets/1b84b0cd-07ff-48f7-bf06-9994c39a23ae" />


I utilized the embedding vectors in the different classification models. I also used kmeans clustering on the embedding vectors in order to group similar occupation vectors together. These clusters were utilized as a feature in the model as well. Below is a look at the vectors as represented by their first two principal components, colored by 7 cluster groups.

<img width="647" alt="amazing_race_principle_components_viz" src="https://github.com/user-attachments/assets/2ca65cf4-8438-4a1d-9200-4846f23e8cf2" />

The first two principal components only accounted for about 16% of the variability in the variables. And we can see above that the clusters of these components don’t seem to have any strong boundaries. 

## NLP - BERT
I also took a crack at converting occupation to embeddings using transformers instead of Word2Vec. In this situation, there was no need to use the OOH to train a model to generate the embeddings. I just used a pre-trained transformer model to convert the text to embeddings.

In both situations, the final embeddings can be used as features in the models.

<img width="633" alt="amazing_race_model_results-2" src="https://github.com/user-attachments/assets/4635fb13-59d9-4462-8532-c74bed7b4a6f" />
<img width="632" alt="amazing_race_model_results-3" src="https://github.com/user-attachments/assets/e6b8f384-48d6-49fd-9d55-53e6a1dc18f6" />

We see that adding occupation features is hit or miss in terms of improving model performance depending on the method for balancing the dataset. When we look at average model results across cross-validation, we see that the performance between different random forest models for instance, is not huge. The high performers are all around

F1: .4

Recall: .55 - .56

Precision: .55 - .63

I was surprised to see that BERT did not have any of the highest cross validated scores for its random forest model because transformers are an easy, yet powerful way to glean understanding from text. But perhaps the specialization of the OOH added some value that the pre-trained transformer did not capture.

I used the random forest model to look at feature importance. The top 5 features used in the model were:

Winners: [('Avg_Age', 0.202), ('Age_x', 0.163), ('Age_y', 0.147), ('Age_Diff', 0.135), ('Female_team', 0.073)]

Top3:[('Age_y', 0.193), ('Age_x', 0.186), ('Avg_Age', 0.184), ('Age_Diff', 0.123), ('Same_State', 0.048)]

## Final Thoughts
Looking at the results above, it’s safe to say that predicting teams is a very challenging task. With limited and imbalanced data, accurate predictions that both capture winners (recall) but also do not identify a bunch of non-winners as winners (precision) is hard to come by. However, on average (cross validation) using our best models we have a little more than a 50% chance of getting both metrics correct

We tried utilizing text data as features to improve the model, and it did in many instances with predicting winning teams for some of the models and resampling methods. This shows the possible value of wringing semantic meaning out of words, and thus how NLP techniques like Word2Vec can be helpful in accomplishing this. But it is worth mentioning that more time and effort still can be put into making these word representations and categorizations more meaningful.

Finally, if I were to move forward as a betting woman, I would put my energy into improving the model that predicts if a team will make it to the top 3 teams in the race. The strongest model here, the random forest, had an average recall of ~60%, meaning that it correctly captured 60% of the teams that made it to the top 3. Of course there is room for improvement, but I think these numbers are encouraging considering the challenges we were working with.

In the future, I hope to harness more text data that I scraped about the contestants and NLP methods to see if it could improve our predictions.
