import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lime.lime_tabular import LimeTabularExplainer
from matplotlib.animation import FuncAnimation

# Load and preprocess the dataset
@st.cache_data  # Streamlit's magic to keep things fast. Cache it or crash it!
def load_data():
    # Load the dataset – where happiness and money collide in the data universe.
    data = pd.read_csv("Money_vs_Happiness_dataset.csv")
    
    # Columns we’ll trust to be numeric – because no one likes surprises here.
    numerical_columns = [
        'Year', 'Life Ladder', 'Log GDP per capita', 'Social support',
        'Healthy life expectancy at birth', 'Freedom to make life choices',
        'Generosity', 'Perceptions of corruption', 'Positive affect',
        'Negative affect', 'Democracy_Index', 'Total_Emissions',
        'Human Development Index', 'Rule_of_Law_Index', 'Median Age',
        'Urban Population (%)', 'Tax_Revenue'
    ]
    
    # Convert to numbers where possible; force the weird stuff to NaN. Nobody has time for rogue strings.
    for col in numerical_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Drop rows with invalid data – it’s not you, it’s your bad data.
    data = data.dropna(subset=numerical_columns)
    
    return data  # Return the cleaned-up dataset, shiny and ready.

# Load data
data = load_data()

# Title and Intro
st.title("🐺 The Wolf of Happiness Street: Does Money Buy Happiness?")
st.markdown("""
Welcome to **The Wolf of Happiness Street**, where we follow Jordan Belfort's journey to uncover life's most debated question:  
*"Does money really buy happiness?"*  
With data, models, and a sprinkle of fun, we dive into the glitz, the grind, and the (sometimes surprising) truths.  
""")

# Sidebar Filters
st.sidebar.header("Filters")
st.sidebar.markdown("**Customize Jordan's world:**")
country_list = sorted(data['Country'].unique())  # Let's keep things tidy, alphabetical FTW.
selected_countries = st.sidebar.multiselect(
    "🌍 Choose Countries for Jordan's Journey:", options=country_list, default=["India", "United States", "Finland"]
)
year_range = st.sidebar.slider(
    "📆 Set the Timeline for Jordan's Pursuit:", int(data['Year'].min()), int(data['Year'].max()), (2010, 2020)
)

# Filter Data
filtered_data = data[
    (data['Country'].isin(selected_countries)) &
    (data['Year'] >= year_range[0]) &
    (data['Year'] <= year_range[1])
]

# Main Dashboard
st.header("📊 Explore Money and Happiness")

# Section 1: Feature Explanations
st.header("🔍 The Cast of Characters (Features)")
st.markdown("""
Here are the key players in Jordan’s story. These features reflect various aspects of wealth, power, and happiness.  
Each has a role in the pursuit of the ultimate question.  
""")

# Playfully explain each feature
engineered_features = {
    "Freedom_Index": "Freedom Index = `Log GDP per capita` × `Freedom to make life choices`. \
Because what’s wealth without the freedom to enjoy it?",
    "Generosity_Per_Dollar": "Generosity Per Dollar = `Generosity` ÷ `Log GDP per capita`. \
Are the rich really generous or just handing out tips to ease their conscience?",
    "Trust_Factor": "Trust Factor = 1 - `Perceptions of corruption`. \
More trust, less corruption. But can you trust the wolf with your happiness?",
    "Social_Cushion_Index": "Social Cushion Index = `Social support` × `Healthy life expectancy at birth`. \
Because everyone needs a safety net in this rollercoaster of life.",
    "Urban_Stress_Balance": "Urban Stress Balance = `Urban Population (%)` × `Negative affect`. \
Hustle culture meets city life stress. Who’s thriving? Who’s barely surviving?",
    "Hedonic_Growth_Rate": "Hedonic Growth Rate = `Positive affect` ÷ `Negative affect`. \
Are people smiling more than frowning? Simple math for complex emotions.",
    "Environmental_Bonus": "Environmental Bonus = `Total Emissions` ÷ `Log GDP per capita`. \
Does the pursuit of happiness come at the cost of clean air?",
    "Positivity_Ratio": "Positivity Ratio = `Positive affect` ÷ (`Negative affect` + 1e-6). \
Are good vibes overpowering the bad ones? Balance is everything.",
    "Trade_Off_Index": "Trade-Off Index = `Log GDP per capita` ÷ `Life Ladder`. \
How much happiness are you squeezing out of every dollar?"
}

# Display feature details
for feature, description in engineered_features.items():
    st.markdown(f"**{feature}:** {description}")

st.markdown("""
Now that we know the cast, let’s see who’s really driving the story!  
""")

# Section 2: Correlation with Happiness
st.header("📈 Who's Helping Jordan Find Happiness?")
st.markdown("""
Let’s uncover which features are most strongly linked to happiness (`Life Ladder`).  
Who’s the real MVP in Jordan’s quest? 💪
""")

# Compute correlations
numeric_data = filtered_data.select_dtypes(include=['float64', 'int64'])
happiness_corr = numeric_data.corr()['Life Ladder'].sort_values(ascending=False)

# Plot correlation
fig, ax = plt.subplots(figsize=(8, 6))
happiness_corr.drop('Life Ladder').plot(kind='bar', ax=ax, color='teal', edgecolor='black')
ax.set_title("Correlation with Happiness (Life Ladder)", fontsize=16)
ax.set_ylabel("Correlation Coefficient", fontsize=12)
ax.set_xlabel("Feature", fontsize=12)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

# Show strongest and weakest correlations
st.markdown("### Top Factors Influencing Happiness")
st.write("**Strong Positive Correlations:**")
st.write(happiness_corr[happiness_corr > 0].drop('Life Ladder').head(3))
st.write("**Strong Negative Correlations:**")
st.write(happiness_corr[happiness_corr < 0].head(3))

# Section 3: Money vs Happiness
st.header("💵 Does Money Buy Happiness?")
st.markdown("""
Let’s plot wealth (`Log GDP per capita`) against happiness (`Life Ladder`).  
Does more money really mean more smiles? Let’s find out. 🤔
""")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=filtered_data, 
    x="Log GDP per capita", 
    y="Life Ladder", 
    hue="Country", 
    alpha=0.7, 
    ax=ax
)
ax.set_title("Money vs Happiness", fontsize=16)
ax.set_xlabel("Wealth (Log GDP per capita)", fontsize=12)
ax.set_ylabel("Happiness (Life Ladder)", fontsize=12)
st.pyplot(fig)

# Section 4: Machine Learning Insights
st.header("🤖 Predicting Happiness: Jordan’s AI Partner")
st.markdown("""
Jordan teams up with AI to predict happiness based on the features.  
Let’s see what the AI thinks is most important.  
""")

features = [
    'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption',
    'Positive affect', 'Negative affect', 'Democracy_Index', 'Total_Emissions',
    'Human Development Index', 'Rule_of_Law_Index', 'Median Age', 'Urban Population (%)', 'Tax_Revenue'
]
X = data[features]
y = data['Life Ladder']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis", ax=ax)
ax.set_title("What Matters Most for Happiness?", fontsize=16)
ax.set_xlabel("Importance", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
st.pyplot(fig)

# Section 5: LIME Explanations
st.header("🧠 Explaining AI Predictions with LIME")
st.markdown("""
Breaking down AI’s predictions with **LIME**.  
This shows how each feature contributes to predicting happiness for different instances.  
""")

# Function to create an enhanced LIME animation
def create_enhanced_lime_animation(explainer, model, X_test, features, filename="lime_explanation_enhanced.gif"):
    # Create a mapping for short feature names
    feature_mapping = {
        'Log GDP per capita': 'GDP',
        'Social support': 'Social',
        'Healthy life expectancy at birth': 'Life Exp.',
        'Freedom to make life choices': 'Freedom',
        'Generosity': 'Generosity',
        'Perceptions of corruption': 'Corruption',
        'Positive affect': 'Pos. Affect',
        'Negative affect': 'Neg. Affect',
        'Democracy_Index': 'Democracy',
        'Total_Emissions': 'Emissions',
        'Human Development Index': 'HDI',
        'Rule_of_Law_Index': 'Rule of Law',
        'Median Age': 'Median Age',
        'Urban Population (%)': 'Urban Pop.',
        'Tax_Revenue': 'Tax Revenue'
    }
    short_features = [feature_mapping.get(f, f) for f in features]

    # Select diverse instances for explanation
    X_test_sorted = X_test.assign(Predicted=model.predict(X_test))
    instances = pd.concat([
        X_test_sorted.nsmallest(2, "Predicted"),  # Lowest happiness
        X_test_sorted.nlargest(2, "Predicted"),  # Highest happiness
        X_test_sorted.sample(3, random_state=42)  # Random
    ])
    explanations = [
        explainer.explain_instance(row.values[:-1], model.predict) for _, row in instances.iterrows()
    ]

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Context for each instance
    instance_contexts = [
        "Low Happiness Prediction", "Low Happiness Prediction",
        "High Happiness Prediction", "High Happiness Prediction",
        "Random Sample", "Random Sample", "Random Sample"
    ]

    # Update function for the animation
    def update(idx):
        ax.clear()
        explanation = explanations[idx]
        contributions = explanation.as_list()
        contrib_features = [feature_mapping.get(c[0], c[0]) for c in contributions]
        contrib_values = [c[1] for c in contributions]

        # Sort features by absolute contribution
        sorted_indices = np.argsort(np.abs(contrib_values))[::-1]
        contrib_features = [contrib_features[i] for i in sorted_indices]
        contrib_values = [contrib_values[i] for i in sorted_indices]

        # Positive and negative contributions
        colors = ["green" if val > 0 else "red" for val in contrib_values]

        ax.barh(contrib_features, contrib_values, color=colors, edgecolor="black")
        ax.set_title(f"LIME Explanation for Instance {idx + 1}: {instance_contexts[idx]}\nPredicted Happiness: {model.predict([instances.iloc[idx].values[:-1]])[0]:.2f}",
                     fontsize=16)
        ax.set_xlabel("Contribution to Happiness Prediction", fontsize=12)
        ax.set_ylabel("Feature", fontsize=2)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(explanations), interval=3000, repeat=False)
    ani.save(filename, writer="imagemagick")

# Create the LIME Explainer
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=features,
    class_names=['Happiness'],
    mode='regression'
)

# Generate the enhanced LIME animation
create_enhanced_lime_animation(explainer, model, X_test, features, filename="lime_explanation_enhanced.gif")

# Display the animation in Streamlit
st.subheader("LIME Explanation Animation 🎥")
st.markdown("""
This animation dynamically shows how each feature contributes to individual happiness predictions. 
Instances are explicitly categorized as **low, high, or random happiness predictions**.
""")
st.image("lime_explanation_enhanced.gif", caption="Enhanced LIME Explanation Animation")

# Section 6: Data Table
st.header("🔍 Explore Jordan’s Data")
st.markdown("Here’s the filtered dataset for you to explore!")
st.dataframe(filtered_data)

# Footer
st.markdown("""
---
**Project made with ❤️ and 💸**  
*"Money doesn’t buy happiness, but it could buy the ChatGPT 4 membership to understand it :)!"*
""")
