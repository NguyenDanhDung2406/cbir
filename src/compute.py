import os

root_groundtruth = './dataset/groundtruth'
root_evaluation = './dataset/evaluation'


def compute_AP(pos_set, ranked_list):
    relevant = 0.0
    average_precision = 0.0
    number_retrieve = 0

    for item in ranked_list:
        number_retrieve += 1
        if item not in pos_set:
            continue
        
        relevant += 1
        average_precision += (relevant / number_retrieve)
    
    if relevant == 0:
        return 0.0

    return average_precision / relevant

def compute_mAP(feature_extractor, crop=False):
    if crop:
        path_evaluation = os.path.join(root_evaluation, 'crop')
    else:
        path_evaluation = os.path.join(root_evaluation, 'original')

    path_evaluation = os.path.join(path_evaluation, feature_extractor)

    AP = 0.0
    number_query = 0.0

    for query in os.listdir(path_evaluation):
        good_file_path = os.path.join(root_groundtruth, f'{query[:-4]}_good.txt')
        ok_file_path = os.path.join(root_groundtruth, f'{query[:-4]}_ok.txt')

        with open(good_file_path, 'r') as file:
            good_set = file.read().strip().split('\n')
        
        with open(ok_file_path, 'r') as file:
            ok_set = file.read().strip().split('\n')
            
        # positive set of ground truth = ok_set + good_set
        pos_set = set(ok_set + good_set)

        if not pos_set:
            print(f"Warning: pos_set is empty for query {query}")
            continue

        ranked_file_path = os.path.join(path_evaluation, query)
        with open(ranked_file_path, 'r') as file:
            ranked_list = file.read().strip().split('\n')

        if not ranked_list:
            print(f"Warning: ranked_list is empty for query {query}")
            continue
        
        AP += compute_AP(pos_set, ranked_list)
        number_query += 1
    
    if number_query == 0:
        print("Warning: No valid queries found.")
        return 0.0
    
    return AP / number_query