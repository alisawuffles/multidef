# 349/399 Project

In order to better understand the settings under which the W2MDEF-GS model succeeds and fails, we investigated whether certain attributes of words and atoms are predictive of model performance. The word attributes we considered included the ranking of word frequency, the number of ground-truth definitions of the word, the semantic diversity of ground-truth definitions, and the word embedding norm.  Atom attributes included the atom weight after decomposition and the part of speech of the atom. We used logistic regression with these attributes to predict two different output variables: individual error types and the 0-1 score. For predicting the score, we trained logistic regression to minimize the cross-entropy between the model output and the score (i.e., we treated the non-0/1 score labels as probabilities). We performed 5-fold validation, where atoms belonging to the same word must always be in the same fold. 

We were unable to predict the individual error labels with accuracy above baseline, which suggests the attributes were not good predictors given the scale of data we had available, and demonstrates that definition generation is still a challenging problem. However, the score prediction model predicts the score with 0.48 loss, compared to the 0.53 baseline, using only atom weight as an attribute, which is a significant predictor with p-value < 0.01. We hypothesize that this is because atoms with greater weight are more likely to represent more dominant senses that are easier to define.  In fact, the atoms with the top 10\% in weight have an average score of 0.35, substantially higher than the average score of 0.19 across all atoms.

In a separate analysis of part of speech, we found that adjectives achieve the highest average score. This is likely the result of having the shortest average output definition length, making the redundancy label (the most common syntactical failure mode) rare. (Adjectives have the redundancy label 3\% of the time, compared to 15\% for nouns.) The result is that when they are partially right, they usually land in category I-II and score at least 0.6.

|POS|avg score|avg length|W|E|
|---|---|---|---|---|
|adj|0.39|5.7|0.45|0.16|
|noun|0.30|10.1|0.52|0.10|
|adv|0.32|7.2|0.60|0.20|
|verb|0.31|5.9|0.58|0.17|
