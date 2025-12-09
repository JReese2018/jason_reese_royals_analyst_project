import pandas as pd
import xgboost as xgb
import json

def engineer_features(df):
    """
    Using the dataset provided, this function creates new features that will be used for model creation
    """
    df['first_base_x'] = 63.64
    df['first_base_y'] = 63.64
    ## Distance from batter to first base
    df['distance_batter_to_first_base'] = ((df['first_base_x'] - df['batter_pos_x_at_throw']) ** 2 + (df['first_base_y'] - df['batter_pos_y_at_throw']) ** 2) ** 0.5
    ## Distance from thrower to reciever
    df['distance_thrower_to_reciever'] = ((df['receiver_pos_x'] - df['throw_pos_x']) ** 2 + (df['receiver_pos_y'] - df['throw_pos_y']) ** 2) ** 0.5
    ## Distnace from bounce to receiver
    df['distance_bounce_to_reciever'] = ((df['receiver_pos_x'] - df['bounce_pos_x']) ** 2 + (df['receiver_pos_y'] - df['bounce_pos_y']) ** 2) ** 0.5
    ## Throw Velocity
    df['throw_velocity'] = ((df['throw_velo_x']) ** 2 + (df['throw_velo_y']) ** 2 + (df['throw_velo_z']) ** 2) ** 0.5
    ## Bounce Velocity
    df['bounce_velocity'] = ((df['bounce_velo_x']) ** 2 + (df['bounce_velo_y']) ** 2 + (df['bounce_velo_z']) ** 2) ** 0.5
    ## Batter result binary
    df['batter_out'] = 0
    df.loc[df['batter_result'] == 'out', 'batter_out'] = 1
    return df

def predict_thrower_out(df, batter_velo_at_throw, distance_batter_to_first_base, distance_thrower_to_reciever):
    thrower_avg = df[['thrower_id', 'exchange_time', 'throw_velocity']].groupby(['thrower_id']).mean().reset_index()
    thrower_avg['batter_velo_at_throw'] = batter_velo_at_throw
    thrower_avg['distance_batter_to_first_base'] = distance_batter_to_first_base
    thrower_avg['distance_thrower_to_reciever'] = distance_thrower_to_reciever
    model = xgb.XGBClassifier()
    model.load_model('thrower_model_v2.ubj')
    ## Loading feature names
    with open('thrower_eval_feature_names.json', 'r') as f:
        feature_names = json.load(f)
    out_probabilities = model.predict_proba(thrower_avg[feature_names])[:, 1].tolist()
    out_predictions = model.predict(thrower_avg[feature_names]).tolist()
    out = pd.DataFrame({'Probabilities (%)': out_probabilities, 'Predictions': out_predictions})
    final_out_display = thrower_avg.join(out)
    ## Only getting probabilities above 50%
    final_out_display = final_out_display.loc[final_out_display['Probabilities (%)'] >= 0.50]
    
    final_out_display.loc[final_out_display['Predictions'] == 1, 'Predictions'] = 'Out'
    final_out_display.loc[final_out_display['Predictions'] == 0, 'Predictions'] = 'Safe'
    final_out_display = final_out_display[['thrower_id', 'Probabilities (%)']]
    final_out_display['Probabilities (%)'] = round(final_out_display['Probabilities (%)'] * 100, 2)
    final_out_display = final_out_display.sort_values(['Probabilities (%)'], ascending = False).reset_index()
    final_out_display = final_out_display[['thrower_id', 'Probabilities (%)']]
    final_out_display = final_out_display.rename(columns={'thrower_id' : 'Thrower ID'})
    return final_out_display

def get_thrower_probability(thrower_ID, df, batter_velo_at_throw, distance_batter_to_first_base, distance_thrower_to_reciever):
    thrower_avg = df[['thrower_id', 'exchange_time', 'throw_velocity']].groupby(['thrower_id']).mean().reset_index()
    thrower_avg['batter_velo_at_throw'] = batter_velo_at_throw
    thrower_avg['distance_batter_to_first_base'] = distance_batter_to_first_base
    thrower_avg['distance_thrower_to_reciever'] = distance_thrower_to_reciever
    model = xgb.XGBClassifier()
    model.load_model('thrower_model_v2.ubj')
    ## Loading feature names
    with open('thrower_eval_feature_names.json', 'r') as f:
        feature_names = json.load(f)
    out_probabilities = model.predict_proba(thrower_avg[feature_names])[:, 1].tolist()
    out_predictions = model.predict(thrower_avg[feature_names]).tolist()
    out = pd.DataFrame({'Probabilities (%)': out_probabilities, 'Predictions': out_predictions})
    player_probability = thrower_avg.join(out)
    player_probability.loc[player_probability['Predictions'] == 1, 'Predictions'] = 'Out'
    player_probability.loc[player_probability['Predictions'] == 0, 'Predictions'] = 'Safe'
    player_probability = player_probability[['thrower_id', 'Predictions', 'Probabilities (%)']]
    player_probability = player_probability.loc[player_probability['thrower_id'] == thrower_ID]
    player_probability['Probabilities (%)'] = round(player_probability['Probabilities (%)'] * 100, 2)
    return player_probability

def predict_receiver_out(df, batter_velo_at_throw, distance_bounce_to_reciever, distance_batter_to_first_base, distance_thrower_to_reciever, bounce_velocity):
    receiver_avg = df[['receiver_id', 'receiver_dist_from_1b']].groupby(['receiver_id']).mean().reset_index()
    receiver_avg['batter_velo_at_throw'] = batter_velo_at_throw
    receiver_avg['distance_bounce_to_reciever'] = distance_bounce_to_reciever
    receiver_avg['distance_batter_to_first_base'] = distance_batter_to_first_base
    receiver_avg['distance_thrower_to_reciever'] = distance_thrower_to_reciever
    receiver_avg['bounce_velocity'] = bounce_velocity
    model = xgb.XGBClassifier()
    model.load_model('receiver_model_v6.ubj')
    ## Leading feature names
    with open('receiver_eval_feature_names.json', 'r') as f:
        feature_names = json.load(f)
    out_probabilities = model.predict_proba(receiver_avg[feature_names])[:, 1].tolist()
    out_predictions = model.predict(receiver_avg[feature_names]).tolist()
    out = pd.DataFrame({'probabilities': out_probabilities, 'predictions': out_predictions})
    final_out_display = receiver_avg.join(out)
    ## Only getting probabilities above 50%
    final_out_display = final_out_display.loc[final_out_display['probabilities'] >= 0.50]
    final_out_display = final_out_display[['receiver_id', 'probabilities', 'predictions']]
    final_out_display.loc[final_out_display['predictions'] == 1, 'predictions'] = 'Out'
    final_out_display.loc[final_out_display['predictions'] == 0, 'predictions'] = 'Safe'
    final_out_display['probabilities'] = round(final_out_display['probabilities'] * 100, 2)
    final_out_display = final_out_display.sort_values(['probabilities'], ascending = False).reset_index()
    final_out_display = final_out_display[['receiver_id', 'probabilities']]
    final_out_display = final_out_display.rename(columns={'receiver_id' : 'Receiver ID', 'probabilities' : 'Probabilities (%)'})
    return final_out_display

def get_receiver_probability(receiver_ID, df, batter_velo_at_throw, distance_batter_to_first_base, distance_thrower_to_reciever, distance_bounce_to_reciever, bounce_velocity):
    receiver_avg = df[['receiver_id', 'receiver_dist_from_1b']].groupby(['receiver_id']).mean().reset_index()
    receiver_avg['batter_velo_at_throw'] = batter_velo_at_throw
    receiver_avg['distance_batter_to_first_base'] = distance_batter_to_first_base
    receiver_avg['distance_thrower_to_reciever'] = distance_thrower_to_reciever
    receiver_avg['distance_bounce_to_reciever'] = distance_bounce_to_reciever
    receiver_avg['bounce_velocity'] = bounce_velocity
    model = xgb.XGBClassifier()
    model.load_model('receiver_model_v6.ubj')
    ## Loading feature names
    with open('receiver_eval_feature_names.json', 'r') as f:
        feature_names = json.load(f)
    out_probabilities = model.predict_proba(receiver_avg[feature_names])[:, 1].tolist()
    out_predictions = model.predict(receiver_avg[feature_names]).tolist()
    out = pd.DataFrame({'Probabilities (%)': out_probabilities, 'Predictions': out_predictions})
    player_probability = receiver_avg.join(out)
    player_probability.loc[player_probability['Predictions'] == 1, 'Predictions'] = 'Out'
    player_probability.loc[player_probability['Predictions'] == 0, 'Predictions'] = 'Safe'
    player_probability = player_probability[['receiver_id', 'Predictions', 'Probabilities (%)']]
    player_probability = player_probability.loc[player_probability['receiver_id'] == receiver_ID]
    player_probability['Probabilities (%)'] = round(player_probability['Probabilities (%)'] * 100, 2)
    return player_probability
