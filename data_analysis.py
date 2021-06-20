import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns


dataset = pd.read_csv('./data/GermEval21_Toxic_Train.csv')

dataset.insert(0, 'f_length', np.nan)
dataset.insert(0, 'f_num_exclamation', np.nan)
dataset.insert(0, 'f_num_question', np.nan)
dataset.insert(0, 'f_num_mentions_user', np.nan)
dataset.insert(0, 'f_num_mentions_mod', np.nan)
dataset.insert(0, 'f_contains_url', np.nan)
dataset.insert(0, 'f_num_emojis', np.nan)

for row in dataset.iterrows():
    idx, data = row

    # Feature 1: Sentence length
    dataset.f_length[idx] = len(data.comment_text)

    # Feature 2: Number of exclamation marks
    dataset.f_num_exclamation[idx] = data.comment_text.count('!')

    # Feature 3: Number of question marks
    dataset.f_num_question[idx] = data.comment_text.count('?')

    # Feature 4: Number of @USER mentions
    dataset.f_num_mentions_user[idx] = data.comment_text.count('@USER') + data.comment_text.count('@ USER')

    # Feature 5: Number of @MODERATOR mentions
    dataset.f_num_mentions_mod[idx] = data.comment_text.count('@MODERATOR') + data.comment_text.count('@ MODERATOR')

    # Feature 6: Check if post contains an URL
    dataset.f_contains_url[idx] = int(data.comment_text.count('http') > 0)

    # Feature 7: Number of emojis
    emoticons = re.finditer(u'[\U0001f600-\U0001f650]', data.comment_text)
    dataset.f_num_emojis[idx] = sum(1 for _ in emoticons)


plt.figure(0)
plt.title('Toxic')
sns.pairplot(dataset,
             hue='Sub1_Toxic',
             diag_kind='hist',
             vars=dataset.columns[:7])

plt.figure(1)
plt.title('Engaging')
sns.pairplot(dataset,
             hue='Sub2_Engaging',
             diag_kind='hist',
             vars=dataset.columns[:7])

plt.figure(2)
plt.title('FactClaiming')
sns.pairplot(dataset,
             hue='Sub3_FactClaiming',
             diag_kind='hist',
             vars=dataset.columns[:7])

plt.show()
