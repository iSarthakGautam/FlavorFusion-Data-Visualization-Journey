import streamlit as st
import pandas as pd
import re
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_extras.no_default_selectbox import selectbox

# set page title and favicon
st.set_page_config(
  page_title=
  'FlavorFusion - Uniting Tastes and Tales: A Data Visualization Journey',
  page_icon=':bar_chart:',
  layout='wide',
)

# Custom CSS to adjust the title's position
st.markdown("""
        <style>
            
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 3rem;
                    padding-right: 2rem;
                }
                .stSelectbox {
                margin-bottom: 0.3rem;  /* Adjust margin bottom for select box */
            }
            
                .stPlotlyChart {
                    margin-bottom: 0.3rem;  /* Adjust margin bottom for Plotly charts */
                }
        </style>
        """,
            unsafe_allow_html=True)
st.title('FlavorFusion - Uniting Tastes and Tales')

# Set up a session state
if 'view_dashboards_clicked' not in st.session_state:
  st.session_state.view_dashboards_clicked = False


def ques_1():
  # Define the order of categories
  category_order = [
    'very unlikely', 'unlikely', 'neutral', 'likely', 'very likely'
  ]
  col1, col2 = st.columns([8, 4])

  with col1:
    cuisines_pref_plot = px.histogram(df,
                                      x=[
                                        'indian_food', 'italian_food',
                                        'thai_food', 'persian_food',
                                        'greek_food'
                                      ],
                                      barmode='group')
    cuisines_pref_plot.update_layout(
      legend_title_text="",
      xaxis_title="Inclination",
      yaxis_title="Number of students",
      title="Inclination towards different Cuisines",
      legend=dict(
        orientation="h",
        yanchor="top",
        y=1,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(0,0,0,0)"  # Transparent background
      ))

    # Update x-axis with category order
    cuisines_pref_plot.update_xaxes(categoryorder="array",
                                    categoryarray=category_order)
    st.plotly_chart(cuisines_pref_plot, use_container_width=True)
    cuisines_pref_plot.update_layout(margin=dict(l=0, r=0, t=30,
                                                 b=0))  # Adjust margins

  with col2:
    favCuisine_text = ' '.join(df['fav_cuisine'])

    # Format the WordCloud title similar to Plotly title
    st.markdown(
      '<h5 style="text-align: center; font-weight: normal; font-size: 16px">Favourite Cuisines of College Students</h5>',
      unsafe_allow_html=True)

    # Create a WordCloud object
    favCuisine_wordcloud = WordCloud(width=800,
                                     height=600,
                                     background_color='white',
                                     max_words=50).generate(favCuisine_text)

    # Plot the WordCloud
    plt.imshow(favCuisine_wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axes
    st.pyplot(plt.gcf(),
              use_container_width=True)  # Set to use container width

  coll1, coll2, coll3 = st.columns(3)
  with coll1:
    breakfast = px.histogram(df, x='breakfast')
    breakfast.update_layout(margin=dict(l=0, r=0, t=30, b=0),
                            title="Breakfast Preferences")  # Adjust margins
    st.plotly_chart(breakfast, use_container_width=True)

  with coll2:
    coffee = px.histogram(df, x='coffee')
    coffee.update_layout(margin=dict(l=0, r=0, t=30, b=0),
                         title="Coffee Preferences")  # Adjust margins
    st.plotly_chart(coffee, use_container_width=True)

  with coll3:
    comfortFood_text = ' '.join(df['comfort_food'])

    # Format the WordCloud title similar to Plotly title
    st.markdown(
      '<h5 style="text-align: center; font-weight: normal; font-size: 16px">Comfort Food of College Students</h5>',
      unsafe_allow_html=True)

    # Create a WordCloud object
    comfort_wordcloud = WordCloud(width=800,
                                  height=600,
                                  background_color='white',
                                  max_words=50).generate(comfortFood_text)

    # Plot the WordCloud
    plt.imshow(comfort_wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axes
    st.pyplot(plt.gcf(),
              use_container_width=True)  # Set to use container width


def ques_2():
  category_order = [
    'very unlikely', 'unlikely', 'neutral', 'likely', 'very likely'
  ]
  col1, col2, col3 = st.columns(3)
  with col1:
    weight_hist = px.histogram(df[df.weight_kg > 0],
                               x="weight_kg",
                               color="Gender")
    weight_hist.update_layout(
      margin=dict(l=0, r=0, t=30, b=0),
      title="Weights of College Students",
      xaxis_title="Weight (kg)",
      yaxis_title="Number of students",
      legend_title_text="",
      legend=dict(
        orientation="h",
        yanchor="top",
        y=0.98,
        xanchor="center",
        x=0.8,
        bgcolor="rgba(0,0,0,0)"  # Transparent background
      ))
    st.plotly_chart(weight_hist, use_container_width=True)

  with col2:
    exercise_order = [
      'Once a week', 'Twice or three times per week', 'Everyday'
    ]
    weight_ex_box = px.box(df[df.weight_kg > 0], y="weight_kg", x="exercise")
    weight_ex_box.update_layout(margin=dict(l=0, r=0, t=30, b=0),
                                title="Weight and Exercise",
                                xaxis_title="Exercise Frequency",
                                yaxis_title="Weight (kg)")
    weight_ex_box.update_xaxes(categoryorder='array',
                               categoryarray=exercise_order)
    st.plotly_chart(weight_ex_box, use_container_width=True)

  with col3:
    weight_healthy_feeling = px.scatter(df[df.weight_kg > 0],
                                        y="weight_kg",
                                        x="healthy_feeling",
                                        color="Gender",
                                        height=430)
    weight_healthy_feeling.update_xaxes(categoryorder='category ascending')
    weight_healthy_feeling.update_layout(
      margin=dict(l=0, r=0, t=30, b=0),
      title="How Healthy do College Students feel?",
      xaxis_title=
      "Healthy Feeling on a scale of 1-10 (1: most healthy and 10: least healthy)",
      yaxis_title="Weight (kg)",
      legend_title_text="",
      legend=dict(
        orientation="h",
        yanchor="top",
        y=0.98,
        xanchor="center",
        x=0.15,
        bgcolor="rgba(0,0,0,0)"  # Transparent background
      ))
    st.plotly_chart(weight_healthy_feeling, use_container_width=True)

  col1, col2, col3 = st.columns([2, 4, 4])
  with col1:
    sports_pie = px.pie(df, names='sports', hole=0.5, height=400)
    sports_pie.update_layout(margin=dict(l=0, r=0, t=30, b=0),
                             title="Play Sports?")
    st.plotly_chart(sports_pie, use_container_width=True)

  with col2:
    veggies_day = px.histogram(df, x='veggies_day')
    veggies_day.update_xaxes(categoryorder="array",
                             categoryarray=category_order)
    veggies_day.update_layout(margin=dict(l=0, r=0, t=30, b=0),
                              title="Consumption of Vegetables in a day",
                              xaxis_title="Likelihood",
                              yaxis_title="Number of students")
    st.plotly_chart(veggies_day, use_container_width=True)

  with col3:
    healthy_food = ' '.join(df['healthy_meal'])

    # Format the WordCloud title similar to Plotly title
    st.markdown(
      '<h5 style="text-align: center; font-weight: normal; font-size: 16px">What do College Students consider as a healthy meal?</h5>',
      unsafe_allow_html=True)

    # Create a WordCloud object
    healthy_meal_wordcloud = WordCloud(width=800,
                                       height=500,
                                       background_color='white',
                                       max_words=50).generate(healthy_food)

    # Plot the WordCloud
    plt.imshow(healthy_meal_wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axes
    st.pyplot(plt.gcf(),
              use_container_width=True)  # Set to use container width


def ques_3():
  col1, col2 = st.columns([8, 4])
  with col1:
    income_mapping = {
      '$70,001 to $100,000': '70k - 100k',
      '$50,001 to $70,000': '50k - 70k',
      'higher than $100,000': '> 100k',
      'less than $15,000': '< 15k',
      '$30,001 to $50,000': '30k - 50k',
      '$15,001 to $30,000': '15k - 30k'
    }
    df['income'] = df['income'].map(income_mapping)
    income_eating_out_count = df.groupby(['income', 'eating_out'
                                          ]).size().reset_index(name='count')
    fig = px.bar(income_eating_out_count,
                 x='income',
                 y='count',
                 color='eating_out',
                 barmode='group',
                 title='Eating Out Frequency Across Income Groups',
                 labels={
                   'count': 'Count',
                   'income': 'Income Group',
                   'eating_out': 'Eating Out Frequency'
                 },
                 category_orders={
                   'income': [
                     '> 100k', '70k - 100k', '50k - 70k', '30k - 50k',
                     '15k - 30k', '< 15k'
                   ]
                 })
    st.plotly_chart(fig, use_container_width=True)

  with col2:
    grouped_data = df.groupby(['eating_out', 'pay_meal_out'
                               ]).size().reset_index(name='count')

    # Create a pie chart for each eating out frequency category
    fig_pie_chart = px.pie(
      grouped_data,
      values='count',
      names='pay_meal_out',
      hole=0.3,
      title='Distribution of Dining Budgets by Eating Out Frequency')

    fig_pie_chart.update_traces(
      textinfo='percent+label',
      insidetextfont=dict(size=12))  # You can adjust the font size

    # Remove the legend
    fig_pie_chart.update_layout(showlegend=False)
    st.plotly_chart(fig_pie_chart, use_container_width=True)
  col1, col2 = st.columns([8, 4])
  with col1:
    grouped_data = df.groupby(['eating_out', 'parents_cook'
                               ]).size().reset_index(name='count')
    total_counts = grouped_data.groupby('eating_out')['count'].transform('sum')
    grouped_data['percentage'] = (grouped_data['count'] / total_counts) * 100

    fig_stacked_bar = px.bar(
      grouped_data,
      x='eating_out',
      y='percentage',
      color='parents_cook',
      title=
      'Relationship Between Eating Out and Parents Cooking (100% Stacked)',
      labels={
        'eating_out': 'Eating Out Frequency',
        'percentage': 'Percentage (%)'
      },
      category_orders={
        'eating_out':
        ['Never', '1-2 times', '2-3 times', '3-5 times', 'every day']
      },
      barmode='relative')  # Use relative barmode for 100% stacked
    st.plotly_chart(fig_stacked_bar, use_container_width=True)

  with col2:
    # Create a Sunburst chart
    fig_sunburst = px.sunburst(df,
                               path=['Gender', 'eating_out'],
                               title='Gender and Eating Out Sunburst Chart')
    st.plotly_chart(fig_sunburst, use_container_width=True)


def ques_4():
  col1, col2 = st.columns([4, 4])
  # Define a custom color mapping
  color_mapping = {
    'slim': '#ef553b',
    'just right': '#00cc96',
    'very fit': '#ab63fa',
    'slightly overweight': '#fecb52',
    'overweight': '#ffa15a',
    'i dont think myself in these terms': '#636efa'
  }
  with col1:
    weight_food_counts = df.groupby(['fav_food', 'self_perception_weight'
                                     ]).size().reset_index(name='count')

    # Pivot the data to create a 100% stacked bar chart
    pivoted_data = weight_food_counts.pivot(index='fav_food',
                                            columns='self_perception_weight',
                                            values='count')
    pivoted_data = pivoted_data.div(pivoted_data.sum(axis=1), axis=0) * 100

    # Create a 100% stacked bar chart
    fig_stacked_bar = px.bar(
      pivoted_data,
      x=pivoted_data.index,
      y=pivoted_data.columns,
      title='Distribution of Self-Perceived Weight within Food Choices',
      labels={
        'x': 'Food Choice',
        'y': 'Percentage'
      },
      color_discrete_map=color_mapping,
      category_orders={"self_perception_weight": list(color_mapping.keys())},
      height=600,
    )
    fig_stacked_bar.update_layout(barmode='relative',
                                  yaxis_title="Percentage of students (%)",
                                  xaxis_title="Food students liked")

    st.plotly_chart(fig_stacked_bar, use_container_width=True)
  with col2:
    weight_categories = [
      'slim', 'just right', 'very fit', 'slightly overweight', 'overweight'
    ]
    df["weight"] = pd.to_numeric(df["weight"], errors='coerce')
    # Categorize actual weight data into the defined weight categories
    df['weight_category'] = pd.cut(df['weight'],
                                   bins=[45, 60, 75, 90, 105, 120],
                                   labels=weight_categories)

    # Count the occurrences of each weight category and food choice combination
    weight_food_counts = df.groupby(['fav_food', 'weight_category'
                                     ]).size().reset_index(name='count')

    # Pivot the data to create a 100% stacked bar chart
    pivoted_data = weight_food_counts.pivot(index='fav_food',
                                            columns='weight_category',
                                            values='count')
    pivoted_data = pivoted_data.div(pivoted_data.sum(axis=1), axis=0) * 100

    # Create a 100% stacked bar chart
    fig_stacked_bar = px.bar(
      pivoted_data,
      x=pivoted_data.index,
      y=pivoted_data.columns,
      title='Distribution of Actual Weight Categories within Food Choices',
      labels={
        'x': 'Food Choice',
        'y': 'Percentage'
      },
      
      color_discrete_map=color_mapping,
      height=600)

    fig_stacked_bar.update_layout(
      barmode='relative',
      yaxis_title="Percentage of students (%)",
      xaxis_title="Food students liked"
    ) 
    st.plotly_chart(fig_stacked_bar, use_container_width=True)


def ques_5():
  # Remove 'and' from comfort food names (case insensitive)
  df['comfort_food'] = df['comfort_food'].apply(lambda x: re.sub(r'\s+and\s+', ', ', x, flags=re.IGNORECASE))

  # Remove full stops and strip leading/trailing spaces
  df['comfort_food'] = df['comfort_food'].str.replace('\.', '').str.strip()

  # Convert to uppercase and split by comma
  df_upper = df['comfort_food'].str.upper()
  df['comfort_foods_list'] = df_upper.str.split(r'[,./]')

  # Remove leading and trailing spaces and commas
  df['comfort_foods_list'] = df['comfort_foods_list'].apply(lambda x: [item.strip(', ').strip() for item in x])

  # Remove consecutive commas not followed by a word
  df['comfort_foods_list'] = df['comfort_foods_list'].apply(lambda x: [re.sub(r',+(?!\s*\w)', '', item) for item in x])

  # Remove extra spaces
  df['comfort_foods_list'] = df['comfort_foods_list'].apply(lambda x: [re.sub(r'\s+', ' ', item) for item in x])

  # Remove non-word characters from the beginning and end of comfort foods
  df['comfort_foods_list'] = df['comfort_foods_list'].apply(lambda x: [re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', item) for item in x])

  # Remove blank entries
  df['comfort_foods_list'] = df['comfort_foods_list'].apply(lambda x: [item for item in x if item])


  # Exploding the list of comfort foods
  df_exploded = df.explode('comfort_foods_list')
  
  # Grouping and counting the comfort foods
  comfort_food_counts = df_exploded['comfort_foods_list'].value_counts().reset_index()
  comfort_food_counts.columns = ['comfort_food', 'count']
  
  # Creating a treemap using Plotly Express
  fig = px.treemap(comfort_food_counts.head(20), path=['comfort_food'], values='count',
                  title='Top 20 Comfort Food',
                  hover_data=['count'],
                  labels={'count': 'Count'},
                  color_discrete_sequence=px.colors.qualitative.D3)
  fig.update_traces(textinfo='label+text+value')
  fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))  # Adjust margins
  
  # Displaying the treemap in Streamlit
  st.plotly_chart(fig, use_container_width=True)

  col1, col2, col3 = st.columns(3)
  with col1:
    # Reasons for Ice cream
    comfort_food_reasons = df[df['comfort_food'].str.contains(
      'Ice cream', case=False)]['comfort_food_reasons']
    comfort_food_reasons = comfort_food_reasons.astype(str)
    combined_text = ' '.join(comfort_food_reasons)

    # Generating the word cloud
    wordcloud = WordCloud(width=400, height=200,
                          background_color='white').generate(combined_text)

    # Creating a figure using Plotly Express
    fig = px.imshow(wordcloud, template='plotly_white')
    title = str('Comfort Food Word Cloud for Ice Cream')
    fig.update_layout(title=title)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Displaying the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
  with col2:
    # Reasons for Pizza
    comfort_food_reasons = df[df['comfort_food'].str.contains(
      'Pizza', case=False)]['comfort_food_reasons']
    comfort_food_reasons = comfort_food_reasons.astype(str)
    combined_text = ' '.join(comfort_food_reasons)

    # Generating the word cloud
    wordcloud = WordCloud(width=400, height=200,
                          background_color='white').generate(combined_text)

    # Creating a figure using Plotly Express
    fig = px.imshow(wordcloud, template='plotly_white')
    title = str('Comfort Food Word Cloud for Pizza')
    fig.update_layout(title=title)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Displaying the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

  with col3:
    # Reasons with select box.
    options_series = df_exploded['comfort_foods_list'].unique().tolist()
    options_series.sort()
    # default_option = options_series[34]

    selected_option = selectbox("Select food",options_series, index=34)
    
    comfort_food_reasons =df[df['comfort_food'].str.contains(selected_option, case=False)]['comfort_food_reasons']
    comfort_food_reasons = comfort_food_reasons.astype(str)
    combined_text = ' '.join(comfort_food_reasons)

    # Generating the word cloud
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate(combined_text)

    # Creating a figure using Plotly Express
    fig = px.imshow(wordcloud, template='plotly_white')
    title = str('Comfort Food Word Cloud for: '+selected_option)
    fig.update_layout(title=title)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Displaying the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def ques_6():
  col1, col2 = st.columns(2)
  with col1:
    #Father Education vs GPA boxplot
    education_order = [
      "graduate degree", "college degree", "some college degree",
      "high school degree", "less than high school"
    ]
    fig = px.box(df,
                 x='father_education',
                 y='GPA',
                 color='Gender',
                 title='Distribution of GPA by Father\'s Education Level',
                 labels={
                   'father_education': 'Father\'s Education',
                   'GPA': 'GPA'
                 },
                 category_orders={"father_education": education_order},
                 color_discrete_map={
                   'Male': 'blue',
                   'Female': 'red'
                 })

    # Display the plot in Streamlit
    st.plotly_chart(fig)
  with col2:
    #Mother Education vs GPA boxplot
    education_order = [
      "graduate degree", "college degree", "some college degree",
      "high school degree", "less than high school"
    ]
    fig = px.box(df,
                 x='mother_education',
                 y='GPA',
                 color='Gender',
                 title='Distribution of GPA by Mother\'s Education Level',
                 labels={
                   'father_education': 'Mother\'s Education',
                   'GPA': 'GPA'
                 },
                 category_orders={"mother_education": education_order},
                 color_discrete_map={
                   'Male': 'blue',
                   'Female': 'red'
                 })

    # Display the plot in Streamlit
    st.plotly_chart(fig)

  # Father profession vs GPA
  sorted_data = df.sort_values(by='GPA')
  fig = px.scatter(sorted_data,
                   x='father_profession',
                   y='GPA',
                   title='Scatter Plot of GPA vs. Father Profession',
                   labels={
                     'father_profession': 'Father Profession',
                     'GPA': 'GPA'
                   })

  # Set the height of the y-axis and range
  y_axis_height = 500  # Desired height in pixels
  y_range = [0, 4.5]  # Desired y-axis range
  fig.update_layout(yaxis_fixedrange=True,
                    yaxis=dict(fixedrange=True, range=y_range),
                    height=y_axis_height)

  # Display the plot in Streamlit
  st.plotly_chart(fig, use_container_width=True)

  # Mother profession vs GPA
  sorted_data = df.sort_values(by='GPA')
  fig = px.scatter(sorted_data,
                   x='mother_profession',
                   y='GPA',
                   title='Scatter Plot of GPA vs. Mother Profession',
                   labels={
                     'mother_profession': 'Mother Profession',
                     'GPA': 'GPA'
                   })

  # Set the height of the y-axis and range
  y_axis_height = 550  # Desired height in pixels
  y_range = [0, 4.5]  # Desired y-axis range
  fig.update_layout(yaxis_fixedrange=True,
                    yaxis=dict(fixedrange=True, range=y_range),
                    height=y_axis_height)
  # Display the plot in Streamlit
  st.plotly_chart(fig, use_container_width=True)


# Dashboard content
if st.session_state.view_dashboards_clicked:
  st.empty()
  c1, c2, c3 = st.columns([4, 2, 2])
  ques_options = [
    'What are the cuisine preferences and diets of College Students?',
    'What are the healthy eating and exercising habits of College Students?',
    'Does income impact college students eating out frequency?',
    'How does weight perception influence college students food choices?',
    'What is the comfort food for the students and why?',
    'Does parental Education and Income impact grades?'
  ]
  with c1:
    ques = selectbox('Select a question',
                     ques_options,
                     no_selection_label="Select a question")

  # load data
  df = pd.read_excel('preprocessed_data.xlsx',
                     dtype={'healthy_feeling': 'object'})

  if ques == ques_options[0]:
    ques_1()

  if ques == ques_options[1]:
    ques_2()

  if ques == ques_options[2]:
    ques_3()

  if ques == ques_options[3]:
    ques_4()

  if ques == ques_options[4]:
    ques_5()

  if ques == ques_options[5]:
    ques_6()
else:

  st.markdown("""
      # Welcome to *FlavorFusion* 🍽️
      
      Step into our dynamic dashboard designed to uncover the intriguing world of college students' food choices. Through captivating data visualizations, we delve into various culinary topics.

      As you interact with our dashboard, you'll experience the fusion of **data and creativity**, all meticulously designed to unravel the stories and connections that shape students' gastronomic adventures. From understanding the *emotional allure of comfort food* to exploring how demographics influence dining habits, each visualization offers unique insights into the diverse palates and lifestyles of college students.

      ## Embark on a Culinary Journey

      Welcome to ***FlavorFusion***, where data transforms into narratives and charts become tales. Explore, discover, and relish the journey of understanding the intersection of flavors and stories, all brought to life through the magic of data visualization.

      Let's uncover the *delicious narratives* together!
      """)
  col1, col2, col3 = st.columns([5, 2, 5])
  with col2:
    if st.button('View Dashboards'):
      st.session_state.view_dashboards_clicked = True

  # Add some space before the "Select a question" dropdown
  st.write("")
