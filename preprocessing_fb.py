import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder


# League Name (.*)=(.*)\.fit_transform(.*)  \t$2.fit($3)\n$1 = $2.transform($1)


def label_handler(le_name, clm_name, data, data_raw, make_new = False):
    if le_name != "target":
        if os.path.exists('saved_encoders/' + le_name + '.pkl') and not make_new:
            le_name = joblib.load('saved_encoders/' + le_name + '.pkl')
            if len(data_raw.loc[data_raw[~data_raw[clm_name].isin(data[clm_name].values)].index, clm_name].values) != 0:
                print("These values for " + clm_name + " are new: ", ', '.join(data_raw.loc[data_raw[~data_raw[clm_name].isin(le_name.classes_)].index, clm_name].values))
            data[clm_name] = le_name.transform(data[clm_name])
            data_raw.loc[data_raw[~data_raw[clm_name].isin(le_name.classes_)].index, clm_name]  =  'unk'
            data_raw[clm_name] = le_name.transform(data_raw[clm_name])
        else:
            print("Encoder for ", le_name, " not found or make_new is True, creating new one.")
            path = 'saved_encoders/' + le_name + '.pkl'
            le_name = LabelEncoder()
            le_name.fit((np.hstack((data[clm_name].values, 'unk'))))
            if len(data_raw.loc[data_raw[~data_raw[clm_name].isin(data[clm_name].values)].index, clm_name].values) != 0:
                print("These values for " + clm_name + " are new: ", ', '.join(data_raw.loc[data_raw[~data_raw[clm_name].isin(le_name.classes_)].index, clm_name].values))
            data_raw.loc[data_raw[~data_raw[clm_name].isin(le_name.classes_)].index, clm_name]  =  'unk'
            data[clm_name] = le_name.transform(data[clm_name])
            data_raw[clm_name] = le_name.transform(data_raw[clm_name])
            joblib.dump(le_name, path)
            
        return le_name, data, data_raw

    else:
        
        if os.path.exists('saved_encoders/target.pkl') and not make_new:
            target = joblib.load('saved_encoders/target.pkl')
            data['target'] = target.transform(data[clm_name])
        else:
            target = LabelEncoder()
            target.fit(data[clm_name])
            data['target'] = target.transform(data[clm_name])
            joblib.dump(target, "saved_encoders/target.pkl")
            
        return target, data, data_raw

        
    
def data_preprocessor(data, data_raw):
    data_raw.rename(columns={'HomeTeam':'Hometeam','Away':'away odds','Home':'home odds','Draw':'draw odds','LeaugeName':'leagName'}, inplace=True)
    data.drop(data[data['Result H/A/D'] == "-"].index, inplace=True)
    data.drop(['Stake','Unnamed: 0','Match','Time','correct scoredelete','halfTime','fullTime', 'Even','First team to Score','Odd','Odd/Even'], axis=1, inplace=True)
    data.fillna(data.mean(), inplace=True)
    data_raw.fillna(data_raw.mean(), inplace=True)
    return data, data_raw

making changes to git repo and learning git ^^

