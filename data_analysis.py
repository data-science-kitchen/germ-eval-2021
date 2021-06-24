import matplotlib.pyplot as plt
from features import *
import pandas as pd
import seaborn as sns
from tqdm import tqdm


dataset = pd.read_csv('./data/GermEval21_Toxic_Train.csv')


#features = [
#    log_num_characters, log_average_word_length, log_word_length_std, positive_sentiment_logits,
#    negative_sentiment_logits, neutral_sentiment_logits
#]
features = [
    log_num_characters, log_average_word_length, spellcheck_quality
]

for feature in features:
    dataset.insert(0, feature.__name__, np.nan)

for row in tqdm(dataset.iterrows(), total=len(dataset)):
    idx, data = row

    for feature in features:
        dataset.loc[idx, feature.__name__] = feature(data.comment_text)


plt.figure(0)
plt.title('Toxic')
sns.pairplot(dataset,
             hue='Sub1_Toxic',
             vars=dataset.columns[:len(features)])

plt.figure(1)
plt.title('Engaging')
sns.pairplot(dataset,
             hue='Sub2_Engaging',
             vars=dataset.columns[:len(features)])

plt.figure(2)
plt.title('FactClaiming')
sns.pairplot(dataset,
             hue='Sub3_FactClaiming',
             vars=dataset.columns[:len(features)])

plt.show()
