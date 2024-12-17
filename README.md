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

To deal with the imbalanced data, I used three methods of resampling to compare model training results: random undersampling, random oversampling, and SMOTE. I also used kfold cross validation to try to mitigate bias in the small training set. However, I saw that with kfold, testing on our hold-out data performed leaps and bounds better than on our regular test set. This indicates that we need to be vigilant about building models on data samples where the underlying distribution does not reflect the unmanipulated data. Here are the model performance results:

<img width="728" alt="amazing_race_model_results" src="https://github.com/user-attachments/assets/c67fe2ef-f572-4083-89d0-21d475dba52c" />

_train/test split random state: 122. Note: This table reflects predicting top3, not Winner. Need to update this.
Could run these results a few more times with different random states just to check how consistent_

Because the data are imbalanced, I didn’t bother looking at accuracy as a performance measure. I focused on recall which tells us out of all the teams that actually won, what percent of those teams did the model predict as winners. We see that the undersampled Random Forest model performed the best, with a recall of .733.

What we see in the results is that the resampling method plays a key part in the model training. Undersampling seems to give more freedom to predict trues for winners, where the opposite is true for oversampling (although this does better than no resampling at all). And since the precision on these undersampled data are the same or better (about a third of the teams we predicted as winners were actually winners), it seems that this may be the best method for training this model moving forward. 

### Possible Improvements
* Ensembling
* Threshold cutoffs
* Feature Engineering

I also wanted to see if ensembling the models would yield better results, where some models may do better on certain cases than others. To do this, I averaged the predicted probabilities of several models togethers, and checked to see if that avg was below or above .5 to make a final prediction of Win or not win. This method did not yield much improved results. 

I did notice though, that lowering the probability cutoff of this ensemble model to .30 resulted in performance metrics similar to undersampled XGBoost. More investigation into probability thresholds could be another avenue of improving model performance.

The next thing that I wanted to investigate was using NLP to layer in some of our text data, such as occupations, as features into the model.

## NLP - Feature Engineering with Word2Vec
I wanted to use occupations in a model because I felt that they could be a proxy for some underlying skills or characteristics of an individual. For example, an engineer may be good at solving problems and a dancer may be agile/athletic. I felt that if I could transform these occupation strings into something that held some intrinsic meaning behind what the job was, this could possibly be a great feature in the classification model.

I immediately thought of Word2Vec. It is a model that converts tokens (words) into embeddings (vectors) by taking the weights from a continuous bag of words (CBOW) model. A CBOW model aims to predict a word in the midst of a window of other words. At first I thought I would train my own model, but I quickly realized that I didn’t have enough data or context around the words that I had to make any meaningful embeddings. I decided to fine-tune a pre-trained Word2Vec model – the Google News vectors.
Plotting a sample of my occupation word vectors into 2D space using t-SNE, this is what I got. You can see that words like doctor, pharmacist, and orthopedic cluster together, while words like dispatcher, trucker, motorcyclist are close together. This gave me some confidence that the word embeddings had picked up some semantic meaning.

<img width="626" alt="amazing_race_embedding_viz" src="https://github.com/user-attachments/assets/1b84b0cd-07ff-48f7-bf06-9994c39a23ae" />


I utilized the embedding vectors in the different classification models. I also used kmeans clustering on the embedding vectors in order to group similar occupation vectors together. These clusters were utilized as a feature in the model as well. Below is a look at the vectors as represented by their first two principal components, colored by 7 cluster groups.

<img width="647" alt="amazing_race_principle_components_viz" src="https://github.com/user-attachments/assets/2ca65cf4-8438-4a1d-9200-4846f23e8cf2" />

The first two principal components only accounted for about 16% of the variability in the variables. And we can see above that the clusters of these components don’t seem to have any strong boundaries. 
Running the model with just the cluster features and not the embedding features yielded worse results for almost all models and resampling methods.
Running the model with just the embedding features and not the clusters yields about the same results as when clusters are included. But definitely no improvement.

Winning Teams as Target

<img width="625" alt="amazing_race_results_forWinners" src="https://github.com/user-attachments/assets/b0ff619f-857b-410a-88dc-0b0a075217f5" />


We see that adding occupation features increase overall performance for the undersampled decision tree and random forest models, but not for xgboost. Still, having precision below 20%, these models aren’t very encouraging.

I wondered if we would have a better time predicting which teams make it to the Top 3, as there may be an element of random luck to who wins when it gets to this level of the race. Here are the performance results for models predicting whether a team made it to the top three.

Top 3 Teams as Target

<img width="625" alt="amazing_race_results_forTop3" src="https://github.com/user-attachments/assets/b5855216-d636-49aa-aa30-6c6145ef9234" />



Here we see that adding occupation as a feature increased model performance where the data were not resampled. However, the undersampled models have yielded the highest promise in the past, and each of these performed worse overall. In most instances, precision is always lower. Perhaps these features imply a relationship that isn’t actually there. We could look at the splits/decisions in the decision tree to get a better understanding of what rule funnels more cases as wins.

I used the random forest model to look at feature importance. The top 5 features used in the model were:

Winners: [('Avg_Age', 0.202), ('Age_x', 0.163), ('Age_y', 0.147), ('Age_Diff', 0.135), ('Female_team', 0.073)]

Top3:[('Age_y', 0.193), ('Age_x', 0.186), ('Avg_Age', 0.184), ('Age_Diff', 0.123), ('Same_State', 0.048)]

## Final Thoughts
Looking at the results above, it’s safe to say that predicting teams is a very challenging task. With limited and imbalanced data, accurate predictions that both capture winners (recall) but also do not identify a bunch of non-winners as winners (precision) is hard to come by.

We tried utilizing text data as features to improve the model, and it did in many instances with predicting winning teams for some of the models and resampling methods. This shows the possible value of wringing semantic meaning out of words, and thus how NLP techniques like Word2Vec can be helpful in accomplishing this. But it is worth mentioning that more time and effort still can be put into making these word representations and categorizations more meaningful.

Finally, if I were to move forward as a betting woman, I would put my energy into improving the model that predicts if a team will make it to the top 3 teams in the race. The undersampled models here consistently had a precision that hovered around a third, meaning that one third of the predictions that the models made for top 3 teams were actually true. And the strongest model here, the random forest, had a recall of 73%, meaning that it correctly captured 73% of the teams that made it to the top 3. Of course there is room for improvement, but I think these numbers are encouraging considering the challenges we were working with.

In the future, I hope to harness more text data that I scraped about the contestants and NLP methods to see if it could improve our predictions.

