import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

all_resources_list = []
all_attack_types_list = []
all_adaptive_types_list = []
all_roles_list = []
all_factions_list = []
all_attributes_list = []

def create_all_info(champions_json : dict):
    retChampionsDict = {}
    champions_key = []
    champions_name = []
    champions_id = []
    champions_resource = []
    champions_attack_type = []
    champions_adaptive_type = []
    champions_roles = []
    champions_factions = []
    champions_attributes = []
    champions_icon = []
    for champion_key in champions_json:
        curr_champion_json = champions_json[champion_key]
        champions_key.append(curr_champion_json["key"]) 
        champions_name.append(curr_champion_json["name"]) 
        champions_id.append(curr_champion_json["id"]) 
        champions_resource.append(curr_champion_json["resource"])
        champions_attack_type.append(curr_champion_json["attackType"])
        champions_adaptive_type.append(curr_champion_json["adaptiveType"])
        champions_roles.append(curr_champion_json["roles"])
        champions_factions.append(curr_champion_json["faction"])
        champions_attributes.append(curr_champion_json["attributeRatings"])
        champions_icon.append(curr_champion_json["icon"]) 

        if not (curr_champion_json["resource"] in all_resources_list):
            all_resources_list.append(curr_champion_json["resource"])
        if not (curr_champion_json["attackType"] in all_attack_types_list):
            all_attack_types_list.append(curr_champion_json["attackType"])
        if not (curr_champion_json["adaptiveType"] in all_adaptive_types_list):
            all_adaptive_types_list.append(curr_champion_json["adaptiveType"])
        
        for role in curr_champion_json["roles"]:
            if not (role in all_roles_list):
                all_roles_list.append(role)
        if not (curr_champion_json["faction"] in all_factions_list):
            all_factions_list.append(curr_champion_json["faction"])
        for attribute in curr_champion_json["attributeRatings"]:
            if not (attribute in all_attributes_list):
                all_attributes_list.append(attribute)

    retChampionsDict["champion_key"] = champions_key
    retChampionsDict["champion_name"] = champions_name
    retChampionsDict["champion_id"] = champions_id
    retChampionsDict["champion_resource"] = champions_resource
    retChampionsDict["champion_attack_type"] = champions_attack_type
    retChampionsDict["champion_adaptive_type"] = champions_adaptive_type
    retChampionsDict["champion_roles"] = champions_roles
    retChampionsDict["champion_factions"] = champions_factions
    retChampionsDict["champion_attributes"] = champions_attributes
    retChampionsDict["champion_icon"] = champions_icon
    return retChampionsDict


def vectorize_resource(champion_resource):
    retList = []
    for resource in all_resources_list:
        if(champion_resource == resource):
            retList.append(1)
        else:
            retList.append(0)
    return retList

def vectorize_attack_type(champion_attack_type):
    retList = []
    for attack_type in all_attack_types_list:
        if(champion_attack_type == attack_type):
            retList.append(1)
        else:
            retList.append(0)
    return retList

def vectorize_adaptive_type(champion_adaptive_type):
    retList = []
    for adaptive_type in all_adaptive_types_list:
        if(champion_adaptive_type == adaptive_type):
            retList.append(1)
        else:
            retList.append(0)
    return retList

def vectorize_roles(champion_roles):
    retList = []
    for role in all_roles_list:
        # print(role, (role in champion_roles))
        if(role in champion_roles):
            retList.append(1)
        else:
            retList.append(0)
    return retList

def vectorize_attributes(champion_attributes):
    retList = []
    for attribute in all_attributes_list:
        retList.append(champion_attributes[attribute])
    return retList

def vectorize_faction(champion_faction):
    retList = []
    for faction in all_factions_list:
        if(champion_faction == faction):
            retList.append(1)
        else:
            retList.append(0)
    return retList

def quantify_resource(champions_resource):
    retList = []
    for champion_resource in champions_resource:
        retList.append(vectorize_resource(champion_resource))
    return retList

def quantify_attack_type(champions_attack_type):
    retList = []
    for champion_attack_type in champions_attack_type:
        retList.append(vectorize_attack_type(champion_attack_type))
    return retList

def quantify_adaptive_type(champions_adaptive_type):
    retList = []
    for champion_adaptive_type in champions_adaptive_type:
        retList.append(vectorize_adaptive_type(champion_adaptive_type))
    return retList

def quantify_roles(champions_roles):
    retList = []
    for champion_roles in champions_roles:
        retList.append(vectorize_roles(champion_roles))
    return retList

def quantify_factions(champions_factions):
    retList = []
    for champion_factions in champions_factions:
        retList.append(vectorize_faction(champion_factions))
    return retList

def quantify_attributes(champions_attributes):
    retList = []
    for champion_attributes in champions_attributes:
        retList.append(vectorize_attributes(champion_attributes))
    return retList

def quantify_champions(champion_info : pd.DataFrame):
    champions_quantified = []
    champions_key = champion_info["champion_key"]
    champions_name = champion_info["champion_name"]
    champions_resource = champion_info["champion_resource"]
    champions_attack_type = champion_info["champion_attack_type"]
    champions_adaptive_type = champion_info["champion_adaptive_type"]
    champions_roles = champion_info["champion_roles"]
    champions_factions = champion_info["champion_factions"]
    champions_attributes = champion_info["champion_attributes"]

    champions_quantified_resource = quantify_resource(champions_resource)
    champions_quantified_attack_type = quantify_attack_type(champions_attack_type)
    champions_quantified_adaptive_type = quantify_adaptive_type(champions_adaptive_type)
    champions_quantified_roles = quantify_roles(champions_roles)
    champions_quantified_factions = quantify_factions(champions_factions)
    champions_quantified_attributes = quantify_attributes(champions_attributes)
    champions_quantified = (np.concatenate((champions_quantified_resource, champions_quantified_attack_type, 
                          champions_quantified_adaptive_type, champions_quantified_roles, 
                          champions_quantified_factions,champions_quantified_attributes), axis=1))
    # return champions_key, champions_quantified
    return champions_name, champions_quantified

def get_champions_sets(champions : list, champions_info : pd.DataFrame):
    champions_listed = champions_info.loc[(champions_info["champion_name"].isin(champions))]
    champions_not_listed = champions_info.loc[(~champions_info["champion_name"].isin(champions))]
    return champions_listed, champions_not_listed

def calc_champions_euclidean_distance(champions_df : (pd.Series, np.ndarray), others_df : (pd.Series, np.ndarray)):
    e_dists = euclidean_distances(champions_df[1], others_df[1])
    champions_euclidean = list(zip(champions_df[0], e_dists))
    for i in range(len(champions_euclidean)):
        curr_champion, e_dist = champions_euclidean[i]
        euclidean_paired = list(zip(others_df[0], e_dist))
        champions_euclidean[i] = (curr_champion, euclidean_paired)
    return champions_euclidean

def calc_avg_champions_euclidean_distance(champions_df : (pd.Series, np.ndarray), others_df : (pd.Series, np.ndarray)):
    e_dists = euclidean_distances(champions_df[1], others_df[1])
    total = None
    for i in range(len(e_dists)):
        if i == 0:
            total = e_dists[i]
        else:
            total = np.add(total, e_dists[i])
    champions_euclidean = list(zip(others_df[0], total))
    return champions_euclidean

def sort_avg_euclideans(champions_avg_euclidean : list):
    champions_avg_euclidean = sorted(champions_avg_euclidean, key = lambda t: t[1])
    return champions_avg_euclidean

def sort_euclideans(champions_euclidean : list):
    for i in range(len(champions_euclidean)):
        champion_name, e_dists = champions_euclidean[i]
        sorted_e_dist = sorted(e_dists, key = lambda t: t[1])
        champions_euclidean[i] = champion_name, sorted_e_dist
    return champions_euclidean

def create_champion_dataframe(URL : str):
    ret_DF = pd.DataFrame()
    get_request = requests.get(URL)
    if not (get_request.status_code == 200):
        print("GET REQUEST FAILED")
        return ret_DF
    champions_json = get_request.json()
    champions_info = create_all_info(champions_json)
    return champions_info
