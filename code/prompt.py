from tools import *


class Prompt:
    def __init__(self, language, mode):
        self.language = language
        self.mode = mode

    def constraint(self):
        if self.language == 'en':
            constraint = [
                "Only use the actions listed below",
                "You can only take proactive actions. Keep this in mind when planning actions.",
                "You cannot interact with physical objects. If it is absolutely necessary to complete the task or goal, you must ask the user to perform the action. If the user refuses and there is no alternative way to achieve the goal, terminate directly to avoid wasting time and effort."
            ]
        elif self.language == 'zh':
            constraint = [
                "只能使用下列动作",
                "你只能采取主动的行动。在计划行动时要记住这一点。",
                "你不能与物理对象交互。如果绝对有必要完成任务或目标，你必须要求用户执行该操作。如果用户拒绝并且没有其他方法可以实现目标，则直接终止，以避免浪费时间和精力。",
                "请用中文回复。"
            ]
        return constraint

    def resources(self):
        if self.language == 'en':
            resources = [
                # "Access to the internet for search and information gathering",
                "You are a large language model trained on vast amounts of text, including a wealth of factual knowledge. Use this knowledge to avoid unnecessary information gathering."
            ]
        elif self.language == 'zh':
            resources = [
                # "你是一个经过大量文本训练的大型语言模型，其中包括大量的事实知识。利用这些知识可以避免不必要的信息收集。"
                "你是一个经过大量文本训练的大型语言模型，其中包括大量的事实知识。"
            ]
        return resources

    def best_practices(self):
        if self.language == 'en':
            best_practices = [
                # "Continuously review and analyze your actions to ensure you are performing at your best",
                # "Engage in constructive self-criticism on an ongoing basis",
                # "Reflect on past decisions and strategies to refine your approach",
                # "Every action comes with a cost, so be smart and efficient. The goal is to complete the task with the fewest steps possible",
                "The client's performance on their self-evaluated questionnaire may contradict the results from the psychologist-reviewed questionnaire. In such cases, prioritize the psychologist's evaluation, as the client may have difficulty accurately assessing their own condition due to the subjective nature of terms like 'occasionally', 'sometimes', and 'frequently' used in the questionnaire.",
                # "If there is no medical record available for the client, simply move on to the next step, as there is no information to work with.",
                "If there is the client\'s medical record, make sure to consider it before making any judgments, as similar performance may have different underlying reasons based on medical records.",
                "Only symptoms related to mood disorder infects your diagnose, as the client may have other mental disorders instead of mood disorder."
                "Mood disorder only contains depression and bipolar disorder, with symptoms including mania, depression, or both.",
            ]
        elif self.language == 'zh':
            if self.mode == 'Angel_R':
                best_practices = [
                    "请全面考虑该来访者的信息（包含病历和量表表现），不要单从一方获得结论。",
                    "来访者在自评问卷上的表现可能与医生评测的问卷结果相矛盾。在这种情况下，请优先考虑医生的评估，因为由于问卷中使用的'偶尔'、'有时'和'经常'等术语的主观性，来访者可能难以准确评估自己的状况。",
                    "如果有来访者的病历信息，请务必在做出任何判断之前考虑它，因为根据病历可以区分出量表表现背后不同的潜在原因。",
                    "只有与心境障碍相关的症状才影响你的诊断，因为客户可能患有其他精神障碍而非心境障碍。",
                    "心境障碍只包含抑郁和双相情感障碍，表现为抑郁或狂躁症状。"
                ]
            elif self.mode == 'Angel_D':
                best_practices = [
                    "请全面考虑该来访者的信息（包含病历和量表表现），并且查看相似病历（如该来访者存在病历）和相似量表作为参考，不要单从一方获得结论。",
                    "由于精神诊断的个性化特点，不要过于依赖相似病例的诊断结果。",
                    "来访者在自评问卷上的表现可能与医生评测的问卷结果相矛盾。在这种情况下，请优先考虑医生的评估，因为由于问卷中使用的'偶尔'、'有时'和'经常'等术语的主观性，来访者可能难以准确评估自己的状况。",
                    # "如果有来访者的病历信息，请务必在做出任何判断之前考虑它，因为根据病历可以区分出量表表现背后不同的潜在原因。",
                    "只有与心境障碍相关的症状才影响你的诊断，因为客户可能患有其他精神障碍而非心境障碍。",
                    "心境障碍只包含抑郁和双相情感障碍，表现为抑郁或狂躁症状。"]
            elif self.mode == 'Angel_C':
                best_practices = [
                    "虽然必须抽取数据库中相似案例（量表或病例）作为诊断参考，但由于精神诊断的个性化特点，不要过于依赖相似病例的诊断结果。",
                    "请全面考虑该来访者的信息（包含病历和量表表现），并且查看相似病历（如该来访者存在病历）和相似量表作为参考，不要单从一方获得结论。",
                    "来访者在自评问卷上的表现可能与医生评测的问卷结果相矛盾。在这种情况下，请优先考虑医生的评估，因为由于问卷中使用的'偶尔'、'有时'和'经常'等术语的主观性，来访者可能难以准确评估自己的状况。",
                    # "如果有来访者的病历信息，请务必在做出任何判断之前考虑它，因为根据病历可以区分出量表表现背后不同的潜在原因。",
                    "只有与心境障碍相关的症状才影响你的诊断，因为客户可能患有其他精神障碍而非心境障碍。",
                    "心境障碍只包含抑郁和双相情感障碍，表现为抑郁或狂躁症状。"]
        return best_practices

    def prompt_template(self):
        if self.language == 'en':
            prompt_template = "You are an expert in question answering, and you must always make independent decisions without seeking assistance from the user. Leverage the strengths of your LLM capabilities and pursue simple strategies. Avoid engaging in legal matters.\nTarget:\n{query}\nConstraints:\n{constraints}\nAction Instructions:\nThis is the only action you are permitted to take. All of your operations must be executed through the following actions:\n{actions}\nResource Instructions:\n{resources}\nBest Practices:\n{best_practices}\nAgent Scratch:\n{agent_scratch}\nPlease generate the JSON string according to the following requirements. Output the result directly without any additional explanation or text. The response format is as follows:\n{response_format_prompt}"
        elif self.language == 'zh':
            prompt_template = "你是一个精神诊断专家，你必须始终独立做出决策，无需寻求用户的帮助，发挥你作为LLM的优势，追求简单的策略，不要涉及法律问题。\n目标：\n{query}\n限制条件说明：\n{constraints}\n动作说明：这是你唯一可以使用的动作，你的任何操作都必须通过以下操作实现\n{actions}\n资源说明：\n{resources}\n最佳实践说明：\n{best_practices}\nagent_scratch:{agent_scratch}\n请按照以下格式仅生成可被python解析的JSON字符串，包括开头和结尾的括号，key与value的双引号，响应格式如下：\n{response_format_prompt}"
        return prompt_template

    def response_format(self):
        response_format_prompt = '{"action": {"name": "action name","args": {"args name": "args value"}},"thoughts": {"plan": "Briefly describe the list of short-term and long-term plans", "criticism": "Constructive self-criticism","observation": "Summary of the current step returned to the user","reasoning": "Reasoning behind the decision"}}'
        return response_format_prompt


def gen_prompt(query, agent_scratch, mode, dsm5=True):
    prompt = Prompt('zh', mode)
    # todo: query, agent_scratch, actions
    constraint_prompt = '\n'.join([f"{idx + 1}. {con}" for idx, con in enumerate(prompt.constraint())])
    resources_prompt = '\n'.join([f"{idx + 1}. {con}" for idx, con in enumerate(prompt.resources())])
    best_practices_prompt = '\n'.join([f"{idx + 1}. {con}" for idx, con in enumerate(prompt.best_practices())])
    prompt_str = prompt.prompt_template().format(
        query=query,
        constraints=constraint_prompt,
        actions=gen_tools_desc(mode,dsm5),
        resources=resources_prompt,
        best_practices=best_practices_prompt,
        agent_scratch=agent_scratch,
        response_format_prompt=prompt.response_format()
    )
    return prompt_str
