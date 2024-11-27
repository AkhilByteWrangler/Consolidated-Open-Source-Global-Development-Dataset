import pandas as pd
import numpy as np

# Load the dataset – the ultimate mash-up of economics, psychology, and social vibes.
file_path = "Money_vs_Happiness_dataset.csv"  
dataset = pd.read_csv(file_path)

# Feature Engineering Context:
# So, this all started with a quick 10-minute call to my psych professor. 
# Spoiler: it turned into an hour-long lecture on happiness, societal dynamics, and a bit of "you should do grad school!"
# The features below are inspired by that enlightening monologue (worth every minute, honestly).
# The idea is to dive deeper into the intersection of money, happiness, and the human experience. Let’s roll.

# 1. Freedom Index: Blending financial freedom (GDP) with perceived life choice autonomy.
# The professor was adamant: happiness thrives when people feel in control of their lives. Makes sense!
dataset['Freedom_Index'] = (
    dataset['Log GDP per capita'] * dataset['Freedom to make life choices']
)

# 2. Generosity per Dollar: Normalizing generosity by GDP, because just tossing dollars isn’t impressive.
# Turns out, it’s not about how much you have, but how much you’re willing to give. Deep stuff, right?
dataset['Generosity_Per_Dollar'] = (
    dataset['Generosity'] / dataset['Log GDP per capita']
)

# 3. Trust Factor: A blend of trust (or lack of corruption) and life satisfaction.
# The prof called this the "trust glue" of society. Without trust, happiness crumbles faster than my willpower during finals.
dataset['Trust_Factor'] = (
    (1 - dataset['Perceptions of corruption']) * dataset['Life Ladder']
)

# 4. Social Cushion Index: Social support meets happiness – it’s like having a group project partner
# who actually does their part. Life feels safer and better with strong connections.
dataset['Social_Cushion_Index'] = (
    dataset['Social support'] * dataset['Life Ladder']
)

# 5. Urban Stress Balance: Urban living and stress levels squared off here.
# The prof said cities bring more opportunities *and* more anxiety. This feature shows who’s thriving and who’s just surviving.
dataset['Urban_Stress_Balance'] = (
    dataset['Urban Population (%)'] * dataset['Negative affect']
)

# 6. Hedonic Growth Rate: GDP growth and happiness, year-over-year. Inspired by the "Hedonic Treadmill" idea:
# that shiny new GDP might not always make you happier. The prof compared this to chasing grades but never feeling satisfied. Ouch.
dataset['Hedonic_Growth_Rate'] = dataset.groupby('Country')[
    'Log GDP per capita'
].diff() / dataset['Life Ladder']

# 7. Environmental Bonus: Adjusting happiness for environmental damage.
# Because being happy while polluting the planet isn’t cool, no matter how rich you are.
dataset['Environmental_Bonus'] = (
    dataset['Life Ladder'] / (1 + dataset['Total_Emissions'])
)

# 8. Positivity Ratio: The ultimate vibes metric – good affect divided by bad affect.
# Positivity should outweigh negativity, but the professor warned me about "toxic positivity." Balance is key.
dataset['Positivity_Ratio'] = (
    dataset['Positive affect'] / (dataset['Negative affect'] + 1e-6)  # Safety net for zero negativity, rare but possible.
)

# 9. Trade-Off Index: GDP per happiness – the "is it worth it?" metric.
# Prof’s words: "If you’re rich but miserable, what’s the point?" Wise, yet slightly unsettling.
dataset['Trade_Off_Index'] = (
    dataset['Log GDP per capita'] / dataset['Life Ladder']
)

# Save the enhanced dataset. Gotta back up all this brilliance.
dataset.to_csv("Money_vs_Happiness_feature_engineered_dataset.csv", index=False)
print("Feature engineering complete. Enhanced dataset saved as 'Money_vs_Happiness_feature_engineered_dataset.csv'.")
print(dataset.head())
