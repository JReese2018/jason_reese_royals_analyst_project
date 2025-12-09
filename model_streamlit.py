## To run the program locally use the below comment in the terminal(crtl+`)
## streamlit run model_streamlit.py

import streamlit as st
import pandas as pd

from functions import engineer_features, predict_thrower_out, predict_receiver_out, get_thrower_probability, get_receiver_probability

st.set_page_config(page_icon='https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/kc.png&h=200&w=200', page_title="Jason Reese Royals Modeling Project")

## Dataset Configuration
df = pd.read_csv('dataset_2025.csv')
## Feature Engineering
df = engineer_features(df)
thrower_avg = df[['thrower_id', 'exchange_time', 'throw_velocity']].groupby(['thrower_id']).mean().reset_index()
receiver_avg = df[['receiver_id', 'receiver_dist_from_1b']].groupby(['receiver_id']).mean().reset_index()
## Ranks
thrower_avg['exchange_time_rank'] = thrower_avg['exchange_time'].rank()
thrower_avg['throw_velocity_rank'] = thrower_avg['throw_velocity'].rank(ascending=False)
receiver_avg['receiver_dist_from_1b_rank'] = receiver_avg['receiver_dist_from_1b'].rank()

## App Layout
col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.title("Jason Reese's Kansas City Royals Modeling Project Concept App")
with col2:
    st.image('https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/kc.png&h=200&w=200')

thrower_eval, thrower_ranks, errant_receiver_eval, errant_receiver_ranks, model_information = st.tabs(["Thrower Evaluation", "Thrower Ranks", "Errant Receiver Evaluation", "Errant Receiver Ranks", "Model Information"])

with thrower_eval:
    st.subheader('The Thrower Evaluation Model is composed of two categorical features to predict if a thrower has the capability to throw an out based on parameters.')
    st.subheader('1. Controllables')
    st.write('These are things that the thrower can control over and can be trained to improve. Things like how fast they can the ball from the glove to their hand and how fast they can throw the ball.')
    st.subheader('2. Noncontrollables')
    st.write('These are the situational factors that the thrower has to overcome. Things like distance between the thrower and receiver or the speed at which the batter is running would fall under this category.')
    st.subheader('By typing in a players name (or in this case, their ID) this will predict how their probability on converting an Out based on the parameters.')
    ## Means
    batter_velo_at_throw_mean = round(df['batter_velo_at_throw'].mean())
    receiver_dist_from_1b_mean = round(df['receiver_dist_from_1b'].mean())
    distance_batter_to_first_base_mean = round(df['distance_batter_to_first_base'].mean())
    distance_thrower_to_reciever_mean = round(df['distance_thrower_to_reciever'].mean())
    ## Maxes
    batter_velo_at_throw_max = round(df['batter_velo_at_throw'].max())
    receiver_dist_from_1b_max = round(df['receiver_dist_from_1b'].max())
    distance_batter_to_first_base_max = round(df['distance_batter_to_first_base'].max())
    distance_thrower_to_reciever_max = round(df['distance_thrower_to_reciever'].max())

    ## Player ID
    thrower_ID = st.number_input('Thrower ID', min_value=1, value=1, format="%d")
    st.caption('If you need a list of applicable IDs, check out the "Thrower Ranks" page for all the IDs')

    batter_velo_at_throw = st.slider("Batter Velocity at Throw", 5, batter_velo_at_throw_max, batter_velo_at_throw_mean)
    distance_batter_to_first_base = st.slider("Batter Distance from 1st Base", 0, 90, distance_batter_to_first_base_mean)
    distance_thrower_to_reciever = st.slider("Thrower Distance from Receiver", 0, distance_thrower_to_reciever_max, distance_thrower_to_reciever_mean)

    thrower_probability = get_thrower_probability(thrower_ID, df, batter_velo_at_throw, distance_batter_to_first_base, distance_thrower_to_reciever)
    thrower_final_out_display = predict_thrower_out(df, batter_velo_at_throw, distance_batter_to_first_base, distance_thrower_to_reciever)
    
    ## Showing Prediction and Probability
    try:
        thrower_prediction = thrower_probability.iloc[0]['Predictions']
        thrower_probability = thrower_probability.iloc[0]['Probabilities (%)']
        display_thrower_id, display_thrower_prediction, display_thrower_probability = st.columns(3)
        with display_thrower_id:
            st.subheader('Thrower ID:')
            st.header(thrower_ID)
        with display_thrower_prediction:
            st.subheader('Prediction:')
            st.header(thrower_prediction)
        with display_thrower_probability:
            st.subheader('Probaility:')
            st.header(f'{thrower_probability}%')

        st.divider()

        thrower_avg_eval = thrower_avg.loc[thrower_avg['thrower_id'] == thrower_ID].reset_index()
        thrower_avg_exchange_time = thrower_avg_eval.iloc[0]['exchange_time']
        thrower_avg_exchange_time_rank = thrower_avg_eval.iloc[0]['exchange_time_rank']
        thrower_avg_throw_velocity = thrower_avg_eval.iloc[0]['throw_velocity']
        thrower_avg_throw_velocity_rank = thrower_avg_eval.iloc[0]['throw_velocity_rank']

        display_exchange_time, display_throw_velocity = st.columns(2)
        with display_exchange_time:
            st.subheader('Average Exchagne Time:')
            st.header(f'{round(thrower_avg_exchange_time, 2)} (Rank: {round(thrower_avg_exchange_time_rank)})')
        with display_throw_velocity:
            st.subheader('Average Throw Velocity:')
            st.header(f'{round(thrower_avg_throw_velocity, 2)} (Rank: {round(thrower_avg_throw_velocity_rank)})')


        ## df stats
        st.caption(f'There are **{len(thrower_final_out_display)}** players who will likely convert an out on the chosen parameters.')
        st.caption('Top Throwers with parameters:')
        st.table(thrower_final_out_display.head(5))
        st.caption('*Because this model is data-driven, it evaluates any combination of inputs as a valid scenario, even when the selected parameters approach situations that would be unrealistic or physically infeasible in-game.')
        st.caption('In these cases, the model is not asserting that the play is truly possible, but estimating the probability based on the closest pattern it has learned.')
        st.caption('Because of this, the tool should be interpreted as a decision support system rather than a strict simulation.')
        st.caption('More information about this in the "Model Information" page.')
    ## Error Handling
    except:
        st.subheader(f'**{thrower_ID}** does not exsit within this dataset.')

with thrower_ranks:
    thrower_avg = thrower_avg[['thrower_id', 'exchange_time', 'exchange_time_rank', 'throw_velocity' ,'throw_velocity_rank']]
    thrower_avg = thrower_avg.rename(columns={'thrower_id': 'Thrower ID','exchange_time': 'Average Exchange Time', 'exchange_time_rank': 'Exchange Time Rank', 'throw_velocity': 'Average Throw Velocity', 'throw_velocity_rank': 'Throw Velocity Rank'})
    thrower_avg = thrower_avg.set_index('Thrower ID')
    st.header('Thrower Ranks')
    st.dataframe(thrower_avg)

with errant_receiver_eval:
    st.subheader('The Errant Receiver Evaluation Model is composed of two categorical features to predict if a 1st Baseman is capable of catching an errant throw based on parameters.')
    st.subheader('1. Controllables')
    st.write("The most notable metric that can be controlledby the receiver is their distance to the base. Each receiver's distance was averaged to get this")
    st.subheader('2. Noncontrollables')
    st.write('These are the situational factors that the receiver cannot control. Things like distance between the thrower and receiver or the speed at which the batter is running would fall under this category.')
    st.subheader('By typing in a players name (or in this case, their ID) this will predict how their probability on converting an Out based on the parameters.')
    ## Means
    batter_velo_at_throw_mean = round(df['batter_velo_at_throw'].mean())
    distance_bounce_to_reciever_mean = round(df['distance_bounce_to_reciever'].mean())
    distance_batter_to_first_base_mean = round(df['distance_batter_to_first_base'].mean())
    distance_thrower_to_reciever_mean = round(df['distance_thrower_to_reciever'].mean())
    bounce_velocity_mean = round(df['bounce_velocity'].mean())
    ## Maxes
    batter_velo_at_throw_max = round(df['batter_velo_at_throw'].max())
    distance_bounce_to_reciever_max = round(df['distance_bounce_to_reciever'].max())
    distance_batter_to_first_base_max = round(df['distance_batter_to_first_base'].max())
    distance_thrower_to_reciever_max = round(df['distance_thrower_to_reciever'].max())
    bounce_velocity_max = round(df['bounce_velocity'].max())

    ## Player ID
    receiver_ID = st.number_input('Receiver ID', min_value=1, value=1, format="%d")
    st.caption('If you need a list of applicable IDs, check out the "Errant Receiver" page for all the IDs')

    batter_velo_at_throw = st.slider("Batter Velocity At Throw", 5, batter_velo_at_throw_max, batter_velo_at_throw_mean)
    distance_batter_to_first_base = st.slider("Batter Distance From 1st Base", 0, 90, distance_batter_to_first_base_mean)
    distance_thrower_to_reciever = st.slider("Thrower Distance From Receiver", 0, distance_thrower_to_reciever_max, distance_thrower_to_reciever_mean)
    distance_bounce_to_reciever = st.slider("Bounce Distance From Receiver", 0, distance_thrower_to_reciever_max, distance_bounce_to_reciever_mean)
    bounce_velocity = st.slider("Bounce Velocity", 0, 180, bounce_velocity_mean)

    receiver_probability = get_receiver_probability(receiver_ID, df, batter_velo_at_throw, distance_batter_to_first_base, distance_thrower_to_reciever, distance_bounce_to_reciever, bounce_velocity)
    receiver_final_out_display = predict_receiver_out(df, batter_velo_at_throw, distance_bounce_to_reciever, distance_batter_to_first_base, distance_thrower_to_reciever, bounce_velocity)
    
    ## Showing Prediction and Probability
    try:
        receiver_prediction = receiver_probability.iloc[0]['Predictions']
        receiver_probability = receiver_probability.iloc[0]['Probabilities (%)']
        display_receiver_id, display_receiver_prediction, display_receiver_probability = st.columns(3)
        with display_receiver_id:
            st.subheader('Receiver ID:')
            st.header(receiver_ID)
        with display_receiver_prediction:
            st.subheader('Prediction:')
            st.header(receiver_prediction)
        with display_receiver_probability:
            st.subheader('Probaility:')
            st.header(f'{receiver_probability}%')

        st.divider()
        receiver_avg_eval = receiver_avg.loc[receiver_avg['receiver_id'] == receiver_ID].reset_index()
        receiver_avg_receiver_dist_from_1b = receiver_avg_eval.iloc[0]['receiver_dist_from_1b']
        receiver_avg_receiver_dist_from_1b_rank = receiver_avg_eval.iloc[0]['receiver_dist_from_1b_rank']

        display_receiver_dist_from_1b = st.columns(1)
        with display_receiver_dist_from_1b[0]:
            st.subheader('Average Distance From 1B:')
            st.header(f'{round(receiver_avg_receiver_dist_from_1b, 2)} (Rank: {round(receiver_avg_receiver_dist_from_1b_rank)})')


        ## df stats
        st.caption(f'There are **{len(receiver_final_out_display)}** players who will likely convert an out on the chosen parameters.')
        st.caption('Top Throwers with parameters:')
        st.table(receiver_final_out_display.head(5))
        st.caption('*Because this model is data-driven, it evaluates any combination of inputs as a valid scenario, even when the selected parameters approach situations that would be unrealistic or physically infeasible in-game.')
        st.caption('In these cases, the model is not asserting that the play is truly possible, but estimating the probability based on the closest pattern it has learned.')
        st.caption('Because of this, the tool should be interpreted as a decision support system rather than a strict simulation.')
        st.caption('More information about this in the "Model Information" page.')
    ## Error Handling
    except:
        st.subheader(f'**{receiver_ID}** does not exsit within this dataset.')

with errant_receiver_ranks:
    receiver_avg = receiver_avg[['receiver_id', 'receiver_dist_from_1b', 'receiver_dist_from_1b_rank']]
    receiver_avg = receiver_avg.rename(columns={'receiver_id': 'Receiver ID','receiver_dist_from_1b': 'Average Receiver Distance from 1B', 'receiver_dist_from_1b_rank': 'Average Receiver Distance from 1B Rank'})
    receiver_avg = receiver_avg.sort_values(['Average Receiver Distance from 1B Rank'])
    receiver_avg = receiver_avg.set_index('Receiver ID')
    st.header('Errant Receiver Ranks')
    st.dataframe(receiver_avg)

with model_information:
    ## This will have the images, precision scores, and accuracy scores of the models to backup the findings
    st.header('Thrower Model')
    st.write('This model was built with the thought of how physical traits can be the determining factor converting Outs. This model, made with XGBoost, achieved a Precision Score of 98.7% with 84% Accuracy meaning it rarely predicted a runner as safe when an out was actually recorded.')
    st.image('thrower_prediction_matrix.png', caption="Thrower Model Confusion Matrix")
    st.write("Interestingly, 'Exchange Time' and 'Throw Velocity' (the two primary physical traits) were not the most important features. Instead, the distance of the batter to first base was actually the most influential variable, nearly twice as important the second as the second ranked feature, the distance of the thrower from the receiver.")
    st.image('thrower_feature_importance.png', caption="Thrower Model Feature Importance")
    st.write("In practice, this aligns with the logic of the game. The further the ball is from 1st Base, the longer it will take to travel. Identifying players who can minimize that time is the core objective of models like this.")
    st.header('Errant Receiver Model')
    st.write('Although this model had limited data, in comparison to the thrower model, it still achieved a Precision Score of 89.9% with 86% Accuracy. Like the thrower model, it rarely predicted a runner as safe when an out was actually recorded.')
    st.image('receiver_prediction_matrix.png', caption="Errant Receiver Model Confusion Matrix")
    st.write("This model really highlights the importance of the 1st Baseman's distance to 1st Base. It is the most important variable and is also the only variable that is directly controlled by the 1st Baseman in this model.")
    st.write('In practical terms, this suggests that players positioned farther from first base are more likely to struggle with catching errant throws.')
    st.image('receiver_feature_importance.png', caption="Thrower Model Feature Importance")
    st.write("It is important to note that context of the play and reliable data can have a large impact on the outcome which is discussed in the next section.")
    st.header('Notes & Future Considerations')
    st.write("During exploratory analysis, some characteristics emerged that could influence model behavior and interpretation. These are noted below along with potential improvements that could be implemented with additional information.")
    st.write("One concern about this dataset is that did seem to have some questionable data that could have influenced the model into computing some of its answers.")
    st.write("For example, assuming I calculated the first base coordinates correctly, when engineering the distance from the batter to 1st Base, that number should not be more than 90 feet since that is the distance between home plate and 1st base. However, some observed values were larger than 90.")
    st.write("This is a critical consideration as distance to first base was heavily weighted by the model and could disproportionately influence predictions if positional coordinates are misaligned.")
    st.write("Many publicly available baseball positional datasets use the right angle of the 1st Base Line on the X axis and the 3rd Base Line on the y axis making the calculations much more simplified, reducing geometric distortion.")
    st.write('For the purposes of the assessment, I worked on this under the assumption that the data and coordinate system was correct.')
    st.write("An additional contexual feature that could have improved the model on errant throws is knowing whether or not the first baseman had an oppourtunity to catch the ball and maintain contact on the bag after the bounce. This could help isolate whether the outcome was driven by throw accuracy or receiving skill.")