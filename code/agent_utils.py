from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import json
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
import pickle
from operator import itemgetter
import torch
from gpt import *


def check_label(response, answer):
    if response == answer:
        check = 1
    else:
        check = 0
    return response, check


def truncate_after_sentence(text, sentence):
    # Find the last occurrence of the sentence
    last_occurrence = text.rfind(sentence)

    # If the sentence is found, return the portion after it
    if last_occurrence != -1:
        return text[last_occurrence + len(sentence):]
    else:
        return "The sentence was not found in the text."


def get_client_data(client_id):
    file_path = '../../data/data_store.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)  # 加载一行 JSON 数据
            if record.get('id') == client_id:  # 根据 id 匹配
                return record
    return 'No record'


def get_client_diagnose(client_id):
    file_path = '../../data/processed/diagnose_patient.csv'
    df = pd.read_csv(file_path)
    for index, record in df.iterrows():
        if record.get('id') == client_id:  # 根据 id 匹配
            return int(record['mood_disorder']), record['SCID诊断']
    return int(-1), ''


def get_train_structured_record(client_id):
    with open('../../data/structured_record_train.json', 'rb') as file:
        medical_records = json.load(file)
    for record in medical_records:
        if record['id'] == client_id:
            return record


def parse_thoughts(response):
    """
           response: {
               "action": {
                   "name": action name,
                   "args": {
                       "args name": args value
                   }
               },
               "thoughts": {
                   "text": thought,
                   "plan":
                   "criticism":
                   "observation": 当前步骤返回给用户的总结
                   "reasoning":
               }
           }
    """
    try:
        thoughts = response.get("thoughts")
        observation = thoughts.get("observation")
        plan = thoughts.get("plan")
        reasoning = thoughts.get("reasoning")
        criticism = thoughts.get("criticism")
        prompt = f"plan: {plan}\nreasoning: {reasoning}\ncriticism: {criticism}\nobservation: {observation}"
        return prompt
    except Exception as err:
        print("parse thoughts error: {}".format(err))
        return "".format(err)


class EMBEDDER:
    def __init__(self, embedder_name):
        self.embedder_name = embedder_name
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def embed_sequence(self, query):
        # self.model.max_seq_length = 1024
        # embedding = self.model.encode(query, normalize_embeddings=True)

        # use bge-m3 dense retrieval
        embedding = self.model.encode(query,
                                      batch_size=12,
                                      max_length=8192)['dense_vecs']
        # print(embedding)
        return embedding

    # def embed_sequence_in_parallel(self, queries_on_date1):
    #     with multiprocessing.Pool(processes=1) as pool:
    #         candidate_embedding_on_date1 = pool.starmap(self.embed_sequence,
    #                                                     [(q,) for q in queries_on_date1], chunksize=50)
    #     return candidate_embedding_on_date1


def load_dsm5_embeddings():
    with open('../../data/embeddings/symptom_embeddings.pkl', 'rb') as f:
        embedding_list = pickle.load(f)
        return embedding_list


def load_train_set_record_embeddings():
    with open('../../data/embeddings/structured_record_train_embeddings.pkl', 'rb') as f:
        embedding_list = pickle.load(f)
        return embedding_list


def load_existing_record_embedding(id, structured=True):
    if structured:
        with open('../../data/embeddings/structured_record_test_embeddings.pkl', 'rb') as f:
            embedding_list = pickle.load(f)
    else:
        with open('../../data/embeddings/raw_record_test_embeddings.pkl', 'rb') as f:
            embedding_list = pickle.load(f)
    for item in embedding_list:
        if item["id"] == id:
            return item['embedding']
    return ""


def load_existing_similar_records_display(digit_id):
    with open('../../data/similar_cases.json', 'rb') as f:
        similar_records = json.load(f)
    for item in similar_records:
        if item['digit_id'] == digit_id:
            instruction = "将来访者的病历作为query，与数据库中以往案例对比，抽取出top-5相似candidates如下：\n"
            candidate_str = ''
            if item['similar_cases'] == "该来访者无病历，无法匹配数据库中相似病历。":
                return "该来访者无病历，无法匹配数据库中相似病历。"
            for candidate in item['similar_cases']:
                candidate_info = str(candidate['案例信息'])
                candidate_score_str = str(candidate['相似度'])
                candidate_str += ("案例信息：" + candidate_info + "; 诊断结果："
                                  + candidate['诊断结果']
                                  + "该病例与query的相似度为" + candidate_score_str
                                  + '\n')
            return instruction + candidate_str


def load_existing_similar_records_analyzed(digit_id):
    with open('../../data/similar_cases.json', 'rb') as f:
        similar_records = json.load(f)
    for item in similar_records:
        if item['digit_id'] == digit_id:
            instruction = "将来访者的病历作为query，与数据库中以往案例对比，抽取出top-5相似candidates的分析结果如下：\n"
            candidate_str = ''
            if item['similar_cases'] == "该来访者无病历，无法匹配数据库中相似病历。":
                return "该来访者无病历，无法匹配数据库中相似病历。"
            for candidate in item['similar_cases']:
                candidate_id = candidate['案例信息']['id']
                candidate_score_str = str(candidate['相似度'])
                candidate_str += ("candidate id: " + candidate_id + "."
                                  + candidate['诊断结果']
                                  + "该病例与query的相似度为" + candidate_score_str
                                  + "。该candidate与当前query病例相似性在于，" + candidate['案例分析']['相似']
                                  + "不同之处在于，" + candidate['案例分析']['不同'] + '\n')
            return instruction + candidate_str


def load_existing_similar_scales_display(digit_id, data_type='real'):
    if data_type == 'real':
        similar_dir = '../../data/similar_scales_raw.json'
    elif data_type == 'syn':
        similar_dir = '../../dataset_synthesis/tabsyn/synthetic/mood_angels/syn_similar_scales_raw.json'
    with open(similar_dir, 'rb') as f:
        similar_scales = json.load(f)
    for item in similar_scales:
        if item['digit_id'] == digit_id:
            instruction = "将来访者的量表表现作为query，与数据库中以往来访者的量表表现对比，抽取出top-5相似candidates如下：\n"
            candidate_str = ''
            for candidate in item['similar_scales']:
                candidate_info = candidate['案例信息']
                candidate_score_str = str(round(candidate['相似度'], 4))
                candidate_str += ("案例信息: " + str(candidate_info) + ", 诊断结果："
                                  + candidate['诊断结果']
                                  + "该病例与query的相似度为" + candidate_score_str
                                  + "。\n")
            return instruction + candidate_str


def load_existing_similar_scales_analyzed(digit_id, data_type='real'):
    if data_type == 'real':
        similar_dir = '../../data/similar_scales.json'
    elif data_type == 'syn':
        similar_dir = '../../dataset_synthesis/tabsyn/synthetic/mood_angels/syn_similar_scales.json'
    with open(similar_dir, 'rb') as f:
        similar_scales = json.load(f)
    for item in similar_scales:
        if item['digit_id'] == digit_id:
            instruction = "将来访者的量表表现作为query，与数据库中以往来访者的量表表现对比，抽取出top-5相似candidates的分析结果如下：\n"
            candidate_str = ''
            for candidate in item['similar_scales']:
                candidate_id = candidate['案例信息']['id']
                candidate_score_str = str(round(candidate['相似度'], 4))
                candidate_str += ("candidate id: " + candidate_id + "."
                                  + candidate['诊断结果']
                                  + "该病例与query的相似度为" + candidate_score_str
                                  + "。该candidate与当前query量表表现的相似性在于，" + str(candidate['案例分析']['相似'])
                                  + "不同之处在于，" + str(candidate['案例分析']['不同']) + '\n')
            return instruction + candidate_str


def digit_id_to_client_id(digit_id):
    with open('../../data/test_zh.json', 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    for item in test_data:
        if item['digit_id'] == digit_id:
            return item['id']


def get_description_from_list(performance_list):
    total_instruction = "以下是来访者在[自评]和[医生评测]的量表中，高度相关问题的表现，以及统计意义上该问题和心境障碍的pearson相关系数：\n"
    depress_instruction = "抑郁心境相关问题：\n"
    suicide_instruction = "自杀相关问题：\n"
    interest_instruction = "精力与兴趣减退相关问题：\n"
    anxiety_instruction = "焦虑相关问题：\n"
    insomnia_instruction = "失眠相关问题：\n"
    depress_des1 = ""
    depress_des2 = ""
    depress_des3 = ""
    depress_des4 = ""
    depress_des5 = ""
    depress_des6 = ""
    suicide_des1 = ""
    suicide_des2 = ""
    interest_des1 = ""
    interest_des2 = ""
    interest_des3 = ""
    interest_des4 = ""
    anxiety_des1 = ""
    anxiety_des2 = ""
    insomnia_des1 = ""
    insomnia_des2 = ""
    for item in performance_list:
        flag = list(item.keys())[0]
        description = item.get(list(item.keys())[1])
        correlation = item.get(list(item.keys())[2])  # 0.6348
        # 抑郁相关
        if flag == 'hamd_total_score':  # 0.6348
            # description = item.get(item.keys()[1])
            # correlation = item.get(item.keys()[2])
            depress_des1 = "[医生评测]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"
        if flag == 'hama_Q6_score':  # 0.6304
            depress_des2 = "[医生评测]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"
        if flag == 'bprs_Q9_score':  # 0.62
            # description = item.get(item.keys()[1])
            # correlation = item.get(item.keys()[2])
            depress_des3 = "[医生评测]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"
        if flag == 'hamd_Q1_score':
            # description = item.get(item.keys()[1])
            # correlation = item.get(item.keys()[2])  # 0.6021
            depress_des4 = "[医生评测]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"
        if flag == 'phq9_total_score':
            # description = item.get(item.keys()[1])
            # correlation = item.get(item.keys()[2])  # 0.592
            depress_des5 = "[自评]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"
        if flag == 'phq9_Q2_score':
            # description = item.get(item.keys()[1])
            # correlation = item.get(item.keys()[2])  # 0.5006
            depress_des6 = "[自评]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"

        # 自杀相关
        if flag == 'hamd_Q3_score':
            # description = item.get(item.keys()[1])
            # correlation = item.get(item.keys()[2])  # 0.6024
            suicide_des1 = "[医生评测]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"
        if flag == 'phq9_Q9_score':
            # description = item.get(item.keys()[1])
            # correlation = item.get(item.keys()[2])  # 0.5057
            suicide_des2 = "[自评]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"

        # 兴趣相关
        if flag == 'hamd_Q7_score':
            # description = item.get(item.keys()[1])
            # correlation = item.get(item.keys()[2])  # 0.607
            interest_des1 = "[医生评测]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"
        if flag == 'hamd_Q22_score':
            # description = item.get(item.keys()[1])
            # correlation = item.get(item.keys()[2])  # 0.5236
            interest_des2 = "[医生评测]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"
        if flag == 'phq9_Q4_score':
            # description = item.get(item.keys()[1])
            # correlation = item.get(item.keys()[2])  # 0.5097
            interest_des3 = "[自评]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"
        if flag == 'phq9_Q1_score':
            # description = item.get(item.keys()[1])
            # correlation = item.get(item.keys()[2])  # 0.5008
            interest_des4 = "[自评]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"

        # 焦虑相关
        if flag == 'hama_total_score':
            # description = item.get(item.keys()[1])
            # correlation = item.get(item.keys()[2])  # 0.5185
            anxiety_des1 = "[自评]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"
        if flag == 'gad7_total_score':
            # description = item.get(item.keys()[1])
            # correlation = item.get(item.keys()[2])  # 0.5185
            anxiety_des2 = "[自评]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"

        # 失眠相关
        if flag == 'hamd_Q4_score':
            # description = item.get(item.keys()[1])
            # correlation = item.get(item.keys()[2])  # 0.5247
            insomnia_des1 = "[医生评测]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"
        if flag == 'hama_Q4_score':
            insomnia_des2 = "[医生评测]" + description + "该问题和心境障碍的相关系数为" + str(correlation) + "。\n"

    if (
            depress_des1 == "" and depress_des2 == "" and depress_des3 == "" and depress_des4 == "" and depress_des5 == "" and depress_des6 == ""
            and suicide_des1 == "" and suicide_des2 == "" and
            interest_des1 == "" and interest_des2 == "" and interest_des3 == "" and interest_des4 == ""
            and anxiety_des1 == "" and anxiety_des2 == ""
            and insomnia_des1 == "" and insomnia_des2 == ""):
        # 无任何量表信息
        return ""
    return (total_instruction + depress_instruction
            + depress_des1 + depress_des2 + depress_des3 + depress_des4 + depress_des5 + depress_des6
            + suicide_instruction + suicide_des1 + suicide_des2
            + interest_instruction + interest_des1 + interest_des2 + interest_des3 + interest_des4
            + anxiety_instruction + anxiety_des1 + anxiety_des2
            + insomnia_instruction + insomnia_des1 + insomnia_des2
            )


def embed_scale_performances(split, data_type):
    if data_type == 'real':
        save_dir = '../../data/embeddings/scale_' + split + '_embeddings.pkl'
        with open('../../data/' + split + '_zh.json', 'r', encoding='utf-8') as file:
            test_data = json.load(file)
    elif data_type == 'syn':
        save_dir = '../../dataset_synthesis/tabsyn/synthetic/mood_angels/syn_' + split + '_embeddings.pkl'
        with open('../../dataset_synthesis/tabsyn/synthetic/mood_angels/syn_'+split+'.json', 'r') as file:
            test_data = json.load(file)
    embedder = EMBEDDER('bge')
    scale_embedding_list = []
    for item in test_data:
        related_performance_list = item['mood_disorder_related_performance']
        performance = get_description_from_list(related_performance_list)
        if performance != "":  # 有量表结果
            record_embedding = embedder.embed_sequence(performance)
            scale_embedding_list.append({
                "digit_id": item['digit_id'],
                "id": item['id'],
                "scale_performance": performance,
                "embedding": record_embedding})

    # save query embeddings
    with open(save_dir, 'wb') as f:
        pickle.dump(scale_embedding_list, f)
    print(split + ': finish embedding scales.')


def save_all_similar_scales(data_type):
    if data_type == 'real':
        with open('../../data/embeddings/scale_test_embeddings.pkl', 'rb') as f:
            test_scale_embeddings = pickle.load(f)
        with open('../../data/embeddings/scale_train_embeddings.pkl', 'rb') as f:
            train_scale_embeddings = pickle.load(f)
        save_dir = '../../data/similar_scales_raw.json'
    elif data_type == 'syn':
        with open('../../dataset_synthesis/tabsyn/synthetic/mood_angels/syn_test_embeddings.pkl', 'rb') as f:
            test_scale_embeddings = pickle.load(f)
        with open('../../dataset_synthesis/tabsyn/synthetic/mood_angels/syn_train_embeddings.pkl', 'rb') as f:
            train_scale_embeddings = pickle.load(f)
        save_dir = '../../dataset_synthesis/tabsyn/synthetic/mood_angels/syn_similar_scales_raw.pkl'

    candidate_id_list = []
    candidate_performance_list = []
    candidate_diagnose_description_list = []
    all_candidate_embedding_list = []
    for item in train_scale_embeddings:
        candidate_id = item['id']
        candidate_performance = item['scale_performance']
        candidate_embedding = torch.tensor(item['embedding']).to('cuda')
        # simi = test_embedding @ candidate_embedding.T
        mood_disorder, case_diagnose = get_client_diagnose(candidate_id)
        if mood_disorder == -1:
            candidate_diagnose_description = "该来访者暂无精神疾病。"
        elif mood_disorder == 1:
            candidate_diagnose_description = "该患者是心境障碍，具体为" + case_diagnose + "。"
        elif mood_disorder == 0:
            candidate_diagnose_description = "该患者不是心境障碍，具体为" + case_diagnose + "。"
        # temp_info_list.append({
        #     "案例信息": {'id': candidate_id, "量表表现": candidate_performance},
        #     "诊断结果": candidate_diagnose_description
        # })
        candidate_id_list.append(candidate_id)
        candidate_performance_list.append(candidate_performance)
        candidate_diagnose_description_list.append(candidate_diagnose_description)
        all_candidate_embedding_list.append(candidate_embedding)

    candidate_embeddings = torch.stack(all_candidate_embedding_list)
    all_correlation_list = []
    for i in range(0, len(test_scale_embeddings)):
        test_digit_id = test_scale_embeddings[i]['digit_id']
        test_id = test_scale_embeddings[i]['id']
        query_embedding = torch.tensor(test_scale_embeddings[i]['embedding']).to('cuda')
        print('toggling ', i, 'in ', len(test_scale_embeddings), ' scale similarity.')
        current_scale_correlation_list = []
        temp_score_list = torch.matmul(candidate_embeddings, query_embedding.unsqueeze(1)).squeeze().tolist()
        result = [{'案例信息': {'id': candidate_id, '量表表现': candidate_performance},
                   '诊断结果': candidate_diagnose_description,
                   '相似度': score
                   } for candidate_id, candidate_performance, candidate_diagnose_description, score in
                  zip(candidate_id_list, candidate_performance_list, candidate_diagnose_description_list,
                      temp_score_list)]
        result.sort(key=lambda x: x['相似度'], reverse=True)
        retrieved_str = "将量表表现与数据库中以往案例对比，抽取出的top-5相似案例如下：\n"
        retrieved_cases = result[0:5]
        for case in retrieved_cases:
            retrieved_str += str(case) + "\n"
        top_simi_for_current_scale = {
            "digit_id": test_digit_id,
            "id": test_id,
            "similar_scales": retrieved_cases
        }
        all_correlation_list.append(top_simi_for_current_scale)

    with open(save_dir, 'w', encoding='utf-8') as json_file:
        json.dump(all_correlation_list, json_file, ensure_ascii=False, indent=4)


def embed_medical_records(split='test', structured=True):
    if structured:
        with open('../../data/structured_record_' + split + '.json', 'r') as f:
            test_data_list = json.load(f)
    elif not structured:
        with open('../../data/raw_record_' + split + '.json', 'r') as f:
            test_data_list = json.load(f)
    record_embedding_list = []
    embedder = EMBEDDER('bge')
    for i in range(0, len(test_data_list)):
        client_id = test_data_list[i]["id"]
        print('\nToggling client ', client_id, ',', i, ' in ', len(test_data_list))
        if structured:
            record = "患者症状：" + test_data_list[i]["患者症状"] + "\n背景信息：" + test_data_list[i]["背景"]
        else:
            record = test_data_list[i]["medical_records"]
        # print(record)
        record_embedding = embedder.embed_sequence(record)
        record_embedding_list.append({"id": client_id, "embedding": record_embedding})

    # save query embeddings
    if structured:
        save_path = '../../data/embeddings/structured_record_' + split + '_embeddings.pkl'
    else:
        save_path = '../../data/embeddings/raw_record_' + split + '_embeddings.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(record_embedding_list, f)
    print('finish embedding records.')


def get_all_scale_str_for_one_client(item):
    scale_list = ['das', 'gad7', 'hcl32', 'mdq', 'phq9', 'shaps',
                  'bprs', 'hama', 'hamd', 'ymrs']
    description = ""
    for scale in scale_list:
        description += item[scale+'_total_score_description'] + '\n'
    return description


def calculate_record_similarity(id):
    record_embedding = load_existing_record_embedding(id)
    if record_embedding != "":
        train_set_record_embeddings = load_train_set_record_embeddings()
        similarity_list = []
        for train_record_embedding in train_set_record_embeddings:
            simi = record_embedding @ train_record_embedding['embedding'].T
            similarity_list.append(simi)
        symptom_correlation_list = []
        for i in range(len(similarity_list)):
            train_set_record_embeddings[i].pop('embedding')
            candidate_info = train_set_record_embeddings[i]
            mood_disorder, case_diagnose = get_client_diagnose(candidate_info['id'])
            candidate_symptom = get_train_structured_record(candidate_info['id'])
            if mood_disorder == -1:
                continue
            elif mood_disorder == 1:
                candidate_record_description = "该患者是心境障碍，具体为" + case_diagnose + "。"
            elif mood_disorder == 0:
                candidate_record_description = "该患者不是心境障碍，具体为" + case_diagnose + "。"
            symptom_correlation_list.append({
                "案例信息": candidate_symptom,
                "诊断结果": candidate_record_description,
                "相似度": similarity_list[i]
            })
        sorted_correlation_list = sorted(symptom_correlation_list, key=itemgetter('相似度'), reverse=True)
        retrieved_str = "将病历与数据库中以往病历对比，抽取出的top-5相似病例如下：\n"
        retrieved_cases = sorted_correlation_list[0:5]
        for case in retrieved_cases:
            retrieved_str += str(case) + "\n"
        return retrieved_str
    return "该来访者无病历，因此无法抽取相似病历。"


def save_all_similar_records():
    with open('../../data/test_zh.json', 'r') as f:
        test_data_list = json.load(f)
    similarity_json = []
    for i in range(0, len(test_data_list)):
        item = test_data_list[i]
        client_id = item["id"]
        digit_id = item["digit_id"]
        reference_answer = item['mood_disorder']
        if reference_answer == -1:  # 沒有诊断结果
            continue

        print('\nToggling client ', client_id, ',', i, ' in ', len(test_data_list))
        similar_cases_str = calculate_record_similarity(client_id)
        similarity_json.append({
            'digit_id': digit_id,
            'id': client_id,
            'similar_cases': similar_cases_str
        })
    with open('../../data/similar_cases.json', 'w', encoding='utf-8') as json_file:
        json.dump(similarity_json, json_file, ensure_ascii=False, indent=4)


def get_scale_performances(digit_id):
    with open('../../data/test_zh.json', 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    for item in test_data:
        if item['digit_id'] == digit_id:
            related_performance_list = item['mood_disorder_related_performance']
            performance = get_description_from_list(related_performance_list)
            return performance
    return "来访者未填写量表"


def analyze_all_similar_scales(data_type):
    if data_type == 'real':
        similar_dir = '../../data/similar_scales_raw.json'
        save_dir = '../../data/similar_scales_raw.json'
    elif data_type == 'syn':
        similar_dir = '../../dataset_synthesis/tabsyn/synthetic/mood_angels/syn_similar_scales_raw.json'
        save_dir = '../../dataset_synthesis/tabsyn/synthetic/mood_angels/syn_similar_scales.json'
    with open(similar_dir, 'rb') as f:
        scale_data = json.load(f)

    new_scale_data = []
    for i in range(0, len(scale_data)):
        query_case = scale_data[i]
        digit_id = query_case['digit_id']
        id = query_case['id']
        similar_scales = query_case['similar_scales']
        query_performance = get_scale_performances(digit_id)
        new_similar_scales = []
        for candidate in similar_scales:
            candidate_performance = candidate['案例信息']['量表表现']
            candidate_score_str = str(candidate['相似度'])
            instruction_prompt = (
                    "以下是当前来访者（query）的量表表现和数据库中一位以往来访者（candidate）的量表表现，它们使用RAG模型计算出的pearson相关性为"
                    + candidate_score_str
                    + "。现在希望candidate能辅助query的诊断，但不希望candidate的诊断结构造成误导，因此请仅用JSON格式总结两者的相似与不同之处。注意，仅总结相似与不同(相似点和不同点分别总结成一段话)，不要做诊断的结论，因为具体的诊断还需要借助其他信息。例如：\n{\"相似\": \"两者在医生评测的抑郁情绪、自杀倾向、兴趣减退及其他相关问题的评分和相关系数上表现一致，说明在这些方面，两者的症状和表现相似。\",\"不同\": \"在部分自评量表（如PHQ-9总分和GAD-7评分）中，candidate 的评分略高于 query，表明 candidate 可能在某些自评方面略显轻微症状。\"}\n\nquery 量表表现：\n"
                    + query_performance
                    + "candidate量表表现：\n"
                    + candidate_performance)
            messages = [
                {"role": "system", "content": "你是一名精神诊断专家。"},
                {"role": "user", "content": instruction_prompt}
            ]
            gpt_response = DeepSeek().generate(messages)
            response_json = parse_llm_response('gpt', gpt_response)
            while response_json == '':
                print("重新调用llm...")
                gpt_response = DeepSeek().generate(messages)
                response_json = parse_llm_response('gpt', gpt_response)
            candidate.update({"案例分析": response_json})
            new_similar_scales.append(candidate)
            print('analyzed one case!')
        temp_dict = {
            "digit_id": digit_id,
            "id": id,
            "similar_scales": new_similar_scales
        }
        print('finish toggling', i, 'in', len(scale_data))
        new_scale_data.append(temp_dict)
        with open(save_dir, 'w', encoding='utf-8') as json_file:
            json.dump(new_scale_data, json_file, ensure_ascii=False, indent=4)


def structure_client_record(digit_id, record1="", existing=True):
    client_id = digit_id_to_client_id(digit_id)
    # next_step = "接下来你需要检索和来访者症状最相似的诊断标准，请调用retrieve_similar_symptoms。"
    if not existing:
        instruction = '请结构化病历，提取总结患者的症状及背景，逐条总结并用"-"间隔，请仅用JSON格式回复。例如：\n{"患者症状": "- 情绪不稳：患者主诉情绪不稳已有7个月，时而低落，时而烦躁易怒，冲家人发脾气。- 焦虑与紧张：在教室上课时无缘无故出现害怕、紧张、焦虑，感觉其他同学在看自己，要针对自己。- 被害妄想：几周前，路人看过患者时，觉得别人针对自己，要对自己做不好的事情。- 社交障碍：无法与同学好好相处，要求父母在校外租房子住，不与家人交流。- 睡眠问题：睡眠时间短，曾在医院就诊并使用助眠药物。- 情绪波动：情绪波动明显，时而低落，时而烦躁。- 破坏行为：几周前因邻居吵架感到害怕与愤怒，将水果刀抛下窗户。", "背景": "- 患者为18岁男性，学生，学习压力较大，进入高中后成绩下降，自暴自弃。- 患者曾看过心理医生，有所好转，考上大学后问题再度加重。- 家庭关系紧张：患者与父母关系疏远，觉得不能相信父母。- 历史治疗：曾在神经内科就诊，使用助眠药物，睡眠改善。- 当前病情：精神状态差，饮食正常，体力和体重无明显变化。"}\n以下是需要结构化的病例：'
        gpt_response = ask_gpt([
            {"role": "system", "content": "你是一名精神诊断专家。"},
            {"role": "user", "content": instruction + record1}
        ])
        final_reply = parse_llm_response('gpt', gpt_response)
        medical_record_dict = {
            "患者症状": final_reply["患者症状"],
            "背景": final_reply["背景"]
        }
        return medical_record_dict
    elif existing:
        with open('../../data/structured_record_test.json', 'rb') as file:
            medical_records = json.load(file)
        for record in medical_records:
            if record['id'] == client_id:
                return record


def get_raw_record_json(digit_id):
    with open('../../data/raw_record_test.json', 'rb') as file:
        medical_records = json.load(file)
    for record in medical_records:
        if record['digit_id'] == digit_id:
            return record


def analyze_all_similar_cases():
    with open('../../data/similar_cases.json', 'rb') as f:
        case_data = json.load(f)

    new_case_data = []
    for i in range(332, len(case_data)):
        query_case = case_data[i]
        digit_id = query_case['digit_id']
        id = query_case['id']
        similar_cases = query_case['similar_cases']
        query_performance = structure_client_record(digit_id)
        new_similar_cases = []
        if similar_cases == "该来访者无病历，无法匹配数据库中相似病历。":
            new_case_data.append(query_case)
        else:
            similar_cases_list = similar_cases.split('\n')[1:-1]
            for candidate in similar_cases_list:
                candidate = candidate.replace('\'', '\"')
                try:
                    candidate = json.loads(candidate)
                except:
                    continue
                candidate_performance = candidate['案例信息']
                candidate_diagnose = candidate['诊断结果']
                candidate_score_str = str(candidate['相似度'])
                instruction_prompt = (
                        "以下是当前来访者的病例（query）和数据库中一位以往来访者的病例（candidate），它们使用RAG模型计算出的pearson相关性为"
                        + candidate_score_str
                        + "。现在希望candidate能辅助query的诊断，但不希望candidate的诊断结构造成误导，因此请仅用JSON格式总结两者的相似与不同之处。注意，仅总结相似与不同，不要做诊断的结论，因为具体的诊断还需要借助其他信息。例如：\n{'相似': '两者在情绪不稳、社交障碍、幻听与幻觉、妄想、睡眠问题、以及药物治疗方面有很多相似之处，表现出一定的相似症状。','不同': '两者在情绪表现（如情绪波动与情绪低落）、认知问题（query有明确的记忆力减退等表现）、脾胃问题（饮食增减相反）、消极行为（如自残行为）以及体力体重变化方面存在显著差异。'}\n\nquery病例：\n"
                        + str(query_performance)
                        + "candidate病例：\n"
                        + str(candidate_performance))
                messages = [
                    {"role": "system", "content": "你是一名精神诊断专家。"},
                    {"role": "user", "content": instruction_prompt}
                ]
                # GPT().generate(messages)
                gpt_response = ask_gpt(messages)
                response_json = parse_llm_response('gpt', gpt_response)
                while response_json == '':
                    print("重新调用llm...")
                    gpt_response = ask_gpt(messages)
                    response_json = parse_llm_response('gpt', gpt_response)
                candidate.update({"案例分析": response_json})
                new_similar_cases.append(candidate)
            temp_dict = {
                "digit_id": digit_id,
                "id": id,
                "similar_cases": new_similar_cases
            }
            print('finish toggling', i, 'in', len(case_data))
            new_case_data.append(temp_dict)
        with open('../../data/similar_cases.json', 'w', encoding='utf-8') as json_file:
            json.dump(new_case_data, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # record
    # embed_all_structured_records('train')
    # embed_medical_records('test',structured=False)
    # save_all_similar_records()
    # analyze_all_similar_cases()
    # scale
    # embed_scale_performances(split='train', data_type='syn')
    # embed_scale_performances(split='test', data_type='syn')
    # save_all_similar_scales(data_type='syn')
    analyze_all_similar_scales(data_type='syn')
    print()
