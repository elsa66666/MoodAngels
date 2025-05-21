from operator import itemgetter
from typing import Optional
import pandas as pd
import os
from langchain_community.tools.tavily_search import TavilySearchResults
import json
from agent_utils import *
import pickle
import operator

# set proxy
os.environ["HTTP_PROXY"] = "127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "127.0.0.1:7890"


# def get_client_scales(id, question_list):
#     client_data = get_client_data(id)
#     client_performance_str = ''
#     for item in question_list:
#         if '_Q' in item:
#             try:
#                 client_performance_str += item + ' performance:\n' + client_data[str(item + '_description')] + '\n'
#             except:
#                 client_performance_str += ''
#         elif 'total_score' in item:
#             try:
#                 client_performance_str += item + ' performance:\n' + client_data[
#                     str(item.rstrip('score') + 'description')] + '\n'
#             except:
#                 client_performance_str += ''
#     return client_performance_str


def get_client_records(digit_id):
    client_id = digit_id_to_client_id(digit_id)
    with open('../../data/medical_records.json', 'r', encoding='utf-8') as file:
        medical_records = json.load(file)
    next_step1 = "接下来你需要结构化病历内容，请调structure_client_record函数。"
    next_step2 = "该来访者无病历，你只能根据其填写的量表信息做诊断，请调用get_client_scales函数。"
    for record in medical_records:
        if record['id'] == client_id:
            return record['medical_records']
    return ''


def get_scale_performances(digit_id):
    with open('../../data/test_zh.json', 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    for item in test_data:
        if item['digit_id'] == digit_id:
            # selected scales
            related_performance_list = item['mood_disorder_related_performance']
            performance = get_description_from_list(related_performance_list)

            # unselected scales
            # performance = get_all_scale_str_for_one_client(item)
            return performance
    return "来访者未填写量表"


def retrieve_similar_symptoms(record, existing=True, structured=True):
    # next_step = "不要着急在这一步下结论，接下来请检查来访者填写的量表，调用get_scale_performances。"
    if not existing and structured:
        record_str = "患者症状：" + record["患者症状"] + "\n背景信息：" + record["背景"]
        embedder = EMBEDDER('bge')
        record_embedding = embedder.embed_sequence(record_str)
    if existing:
        record_embedding = load_existing_record_embedding(record["id"], structured=structured)
    dsm5_embeddings = load_dsm5_embeddings()
    similarity_list = []
    for dsm5_embedding in dsm5_embeddings:
        simi = record_embedding @ dsm5_embedding['embedding'].T
        similarity_list.append(simi)
    symptom_correlation_list = []
    for i in range(len(similarity_list)):
        dsm5_embeddings[i].pop('embedding')
        symptom_correlation_list.append({
            "症状信息": dsm5_embeddings[i],
            "相关性": similarity_list[i]
        })
    sorted_correlation_list = sorted(symptom_correlation_list, key=itemgetter('相关性'), reverse=True)
    return sorted_correlation_list[0:5]


def previous_cases_display(digit_id, existing=True):
    raw_record = get_client_records(digit_id)
    if raw_record == '':
        return "该来访者无病历，无法匹配数据库中相似病历。"
    else:  # 存在raw record
        if existing:
            retrieved_str = load_existing_similar_records_display(digit_id)
        return retrieved_str


def previous_scales_display(digit_id, existing=True):
    if existing:
        retrieved_str = load_existing_similar_scales_display(digit_id, data_type='real')
        return retrieved_str


def previous_syn_scales_display(digit_id, existing=True):
    if existing:
        retrieved_str = load_existing_similar_scales_display(digit_id, data_type='syn')
        return retrieved_str


def previous_cases_analysis(digit_id, existing=True):
    raw_record = get_client_records(digit_id)
    if raw_record == '':
        return "该来访者无病历，无法匹配数据库中相似病历。"
    else:  # 存在raw record
        if existing:
            retrieved_str = load_existing_similar_records_analyzed(digit_id)
        return retrieved_str


def previous_scales_analysis(digit_id):
    with open('../../data/similar_scales_raw.json', 'rb') as f:
        similar_scales = json.load(f)
    for item in similar_scales:
        if item['digit_id'] == digit_id:
            query_case = item
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
                llm_response = DeepSeek().generate(messages)
                response_json = parse_llm_response('deepseek', llm_response)
                while response_json == '':
                    print("重新调用llm...")
                    llm_response = DeepSeek().generate(messages)
                    response_json = parse_llm_response('deepseek', llm_response)
                candidate.update({"案例分析": response_json})
                new_similar_scales.append(candidate)
            instruction = "将来访者的量表表现作为query，与数据库中以往来访者的量表表现对比，抽取出top-5相似candidates的分析结果如下：\n"
            candidate_str = ''
            for candidate in new_similar_scales:
                candidate_id = candidate['案例信息']['id']
                candidate_score_str = str(round(candidate['相似度'], 4))
                candidate_str += ("candidate id: " + candidate_id + "."
                                  + candidate['诊断结果']
                                  + "该病例与query的相似度为" + candidate_score_str
                                  + "。该candidate与当前query量表表现的相似性在于，" + str(
                            candidate['案例分析']['相似'])
                                  + "不同之处在于，" + str(candidate['案例分析']['不同']) + '\n')
    return instruction+candidate_str


def previous_syn_scales_analysis(digit_id):
    return load_existing_similar_scales_analyzed(digit_id, data_type='syn')


def toggle_client_record(digit_id):
    raw_record = get_client_records(digit_id)
    if raw_record == '':
        return "该来访者无病历，请根据量表信息(及数据库中相似表现的量表辅助)做出判断。"
    else:
        # structured medical record
        # structured_record = structure_client_record(digit_id)
        # similar_symptoms = retrieve_similar_symptoms(structured_record, existing=True, structured=True)
        # client_record_info_str = ("来访者的结构化病历如下：\n"
        #                           + "症状：" + str(structured_record["患者症状"]) + "\n"
        #                           + "背景：" + str(structured_record["背景"]) + "\n"
        #                           + "将病历与DSM5诊断标准对比，抽取出的top-5相关症状如下：\n")

        # unstructured medical record
        raw_record_json = get_raw_record_json(digit_id)
        similar_symptoms = retrieve_similar_symptoms(record=raw_record_json, existing=True, structured=False)
        client_record_info_str = (
                "来访者的病历如下：\n" + raw_record
                + "\n将病历与DSM5诊断标准对比，抽取出的top-5相关症状如下：\n"
        )

        # retrieve in structured, return to agent in unstructured
        # structured_record = structure_client_record(digit_id)
        # similar_symptoms = retrieve_similar_symptoms(structured_record, existing=True, structured=True)
        # client_record_info_str = (
        #         "来访者的病历如下：\n" + raw_record
        #         + "\n将病历与DSM5诊断标准对比，抽取出的top-5相关症状如下：\n"
        # )

        for item in similar_symptoms:
            client_record_info_str += str(item) + "\n"
        return client_record_info_str


def get_scale_performances_syn(digit_id):
    with open('../../dataset_synthesis/tabsyn/synthetic/mood_angels/syn_test.json', 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    for item in test_data:
        if item['digit_id'] == digit_id:
            # selected scales
            related_performance_list = item['mood_disorder_related_performance']
            performance = get_description_from_list(related_performance_list)

            # unselected scales
            # performance = get_all_scale_str_for_one_client(item)
            return performance
    return "来访者未填写量表"


# 这里tool name和“name”对应，函数传入的参数和"args"对应
tools_info = [
    {
        "name": "toggle_client_record",
        "description": "如来访者存在病历信息，该函数将回复结构化的病历信息以及与DSM5诊断标准对比后抽取出的top-5相关症状。",
        "args": [{
            "name": "digit_id",
            "type": "int",
            "description": "来访者id，应在query中提供"
        }]
    }, {
        "name": "get_scale_performances",
        "description": "获取该来访者填写与心境障碍相关性前5%问题的表现，其中各个问题与心境障碍的相关系数代表统计意义上的相关性。",
        "args": [{
            "name": "digit_id",
            "type": "int",
            "description": "来访者id，应在query中提供"
        }]
    }, {
        "name": "previous_cases_display",
        "description": "如来访者存在病历信息，该函数将回复该病历与数据库中以往的结构化病历信息对比后，抽取出的相似性top-5案例。注意精神诊断的个性化特征，抽取出的相似案例仅供参考。",
        "args": [{
            "name": "digit_id",
            "type": "int",
            "description": "来访者id，应在query中提供"
        }]
    }, {
        "name": "previous_scales_display",
        "description": "该函数将回复该来访者的量表表现与数据库中以往的案例量表表现对比后，抽取出的相似性top-5案例。注意精神诊断的个性化特征，抽取出的相似案例仅供参考。",
        "args": [{
            "name": "digit_id",
            "type": "int",
            "description": "来访者id，应在query中提供"
        }]
    }, {
        "name": "previous_cases_analysis",
        "description": "如来访者存在病历信息，该函数将回复结构化的病历信息以及与数据库中以往的结构化病历信息对比后，抽取出的相似性top-5案例。注意精神诊断的个性化特征，抽取出的相似案例仅供参考。",
        "args": [{
            "name": "digit_id",
            "type": "int",
            "description": "来访者id，应在query中提供"
        }]
    }, {
        "name": "previous_scales_analysis",
        "description": "该函数将回复该来访者的量表表现与数据库中以往的案例量表表现对比后，抽取出的相似性top-5案例。注意精神诊断的个性化特征，抽取出的相似案例仅供参考。",
        "args": [{
            "name": "digit_id",
            "type": "int",
            "description": "来访者id，应在query中提供"
        }]
    }, {
        "name": "finish",
        "description": "the whole process is finished",
        "args": [{
            "name": "answer",
            "type": "string",
            "description": "the final answer. Please reply with yes or no"
        }, {
            "name": "reasons",
            "type": "string",
            "description": "conclude your reasons behind the judgement by points"
        }]
    }
]

# 这里是所有可执行的函数构成的列表
tools_map = {
    "toggle_client_record": toggle_client_record,
    "get_scale_performances": get_scale_performances,
    "previous_cases_display": previous_cases_display,
    "previous_scales_display": previous_scales_display,
    "previous_cases_analysis": previous_cases_analysis,
    "previous_scales_analysis": previous_scales_analysis
}


tools_map_syn = {
    "toggle_client_record": toggle_client_record,
    "get_scale_performances": get_scale_performances_syn,
    "previous_cases_display": previous_cases_display,
    "previous_scales_display": previous_syn_scales_display,
    "previous_cases_analysis": previous_cases_analysis,
    "previous_scales_analysis": previous_syn_scales_analysis
}


def gen_tools_desc(mode):
    global tools_info
    tools_desc = []
    if mode == 'raw_agent':
        tools_info = tools_info[0:2] + tools_info[-1:]
    elif mode == 'similar_cases_display':
        tools_info = tools_info[0:4] + tools_info[-1:]
    elif mode == 'similar_cases_analyze':
        tools_info = tools_info[0:2] + tools_info[-3:]
    for idx, t in enumerate(tools_info):
        args_desc = []
        if 'args' in t.keys():
            for info in t['args']:
                args_desc.append({
                    "name": info["name"],
                    "description": info['description'],
                    "type": info["type"]
                })
            args_desc = json.dumps(args_desc, ensure_ascii=False)
            tool_description = f"{idx + 1}. {t['name']}: {t['description']}, args: {args_desc}"
            tools_desc.append(tool_description)
        else:
            tool_description = f"{idx + 1}. {t['name']}: {t['description']}"
            tools_desc.append(tool_description)
    tools_prompt = "\n".join(tools_desc)
    return tools_prompt


def call_llm(model_name, messages, use_alternative=False, retry_num=2):
    trying_num = 0
    if (model_name == 'gpt') and not use_alternative:
        try:
            answer = GPT().generate(messages)
            return answer
        except:
            trying_num += 1
            if trying_num <= retry_num:
                answer = call_llm(model_name, messages, use_alternative=True, retry_num=retry_num)
                return answer
    elif (model_name == 'gpt') and use_alternative:
        try:
            answer = ask_gpt(messages)
            return answer
        except:
            trying_num += 1
            if trying_num <= retry_num:
                answer = call_llm(model_name, messages, use_alternative=True, retry_num=retry_num)
                return answer
    elif (model_name == 'deepseek') and not use_alternative:
        try:
            answer = DeepSeek().generate(messages)
            return answer
        except:
            trying_num += 1
            if trying_num <= retry_num:
                answer = call_llm(model_name, messages, use_alternative=False, retry_num=retry_num)
                return answer


if __name__ == '__main__':
    # get_client_performance('P965', ['phq9_total_score'])
    medical_record_dict = {
        "id": "P995",
        "患者症状": "- 情绪不稳：约一周前在病房照顾家人后开始出现情绪不稳，烦躁易怒。- 头痛：伴有头痛，怕吵。- 恐吵及心慌：听到声音就感到心慌气短。- 口苦口干：自觉嘴里发苦、口干舌燥、没胃口。- 睡眠问题：睡眠差，近三天整夜难以入睡，有时昏睡后会突然惊醒、烦躁。- 思维杂乱：睡不着时会想一些乱七八糟的事情。- 兴趣减退及焦虑：做事提不起兴趣、紧张焦虑。",
        "背景": "- 患者为39岁女性。- 近期在照顾家人中出现情绪与身体症状。- 自行尝试多种药物治疗，包括乌灵胶囊、右佐匹克隆、文拉法辛、舒肝解郁胶囊、曲唑酮等，但症状无明显缓解。- 起病以来精神状态一般，饮食及睡眠差，但二便正常，体力体重未见明显改变。"
    }
    # retrieve_similar_symptoms(medical_record_dict, existing=True)
    # client_record_info_str = toggle_client_record(201376)
    scale_performance = get_scale_performances(201376)
    # test = previous_cases_analysis(201376)
    # test = previous_scales_analysis(101353)
    # gen_tools_desc(mode='raw_agent')
    print()
