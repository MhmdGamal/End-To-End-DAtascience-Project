import streamlit as st
import pandas as pd
import pickle

st.title('Life Expectancy Calculator')




df = pd.read_csv('Life Expectancy Data.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

features = ['diphtheria', 'thinness_5-9_years', 'status', 'measles',
       'income_composition_of_resources', 'bmi', 'infant_deaths',
       'thinness__1-19_years', 'year', 'gdp']

new_df = pd.DataFrame()

new_df['status'] = [st.selectbox('Country Status', df['status'].unique())]
new_df['year'] =[st.selectbox('Year', df['year'].unique())]
# new_df['adult_mortality'] = [st.number_input('adult_mortality', int(df['adult_mortality'].min()), int(df['adult_mortality'].max()))]
new_df['infant_deaths'] = [st.number_input('infant_deaths', int(df['infant_deaths'].min()), int(df['infant_deaths'].max()))]
# new_df['alcohol'] = [st.number_input('alcohol', float(df['alcohol'].min()), float(df['alcohol'].max()))]
# new_df['percentage_expenditure'] = [st.number_input('percentage_expenditure', float(df['percentage_expenditure'].min()), float(df['percentage_expenditure'].max()))]
# new_df['hepatitis_b'] = [st.number_input('hepatitis_b', float(df['hepatitis_b'].min()), float(df['hepatitis_b'].max()))]
new_df['measles'] = [st.number_input('measles', int(df['measles'].min()), int(df['measles'].max()))]
new_df['bmi'] = [st.number_input('bmi', float(df['bmi'].min()), float(df['bmi'].max()))]
# new_df['under-five_deaths'] = [st.number_input('under-five_deaths', int(df['under-five_deaths'].min()), int(df['under-five_deaths'].max()))]
# new_df['polio'] = [st.number_input('polio', float(df['polio'].min()), float(df['polio'].max()))]
# new_df['total_expenditure'] = [st.number_input('total_expenditure', float(df['total_expenditure'].min()), float(df['total_expenditure'].max()))]
new_df['diphtheria'] = [st.number_input('diphtheria', float(df['diphtheria'].min()), float(df['diphtheria'].max()))]
#new_df['hiv/aids'] = [st.number_input('hiv/aids', float(df['hiv/aids'].min()), float(df['hiv/aids'].max()))]
new_df['gdp'] = [st.number_input('gdp', float(df['gdp'].min()), float(df['gdp'].max()))]
new_df['thinness__1-19_years'] = [st.number_input('thinness_1-19_years', float(df['thinness__1-19_years'].min()), float(df['thinness__1-19_years'].max()))]
new_df['thinness_5-9_years'] = [st.number_input('thinness_5-9_years', float(df['thinness_5-9_years'].min()), float(df['thinness_5-9_years'].max()))]
new_df['income_composition_of_resources'] = [st.number_input('income_composition_of_resources', float(df['income_composition_of_resources'].min()), float(df['income_composition_of_resources'].max()))]
#new_df['schooling'] = [st.number_input('schooling', float(df['schooling'].min()), float(df['schooling'].max()))]

st.write('''
## Predicted Life Expectancy
''')
X_new = new_df[features]
transformer = pickle.load(open('transformer.pkl', 'rb'))
X_new = transformer.transform(X_new)
model = pickle.load(open('model.pkl', 'rb'))
prediction = model.predict(X_new)
st.info('''##  {} years'''.format(round(prediction[0], 1)))
