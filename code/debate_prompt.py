class DebatePrompt:
    def __init__(self, type):
        self.type = type

    def constraint(self):
        if self.type in ['positive', 'negative']:
            constraint = [
                "请坚持你的观点，所有需要讨论的案例都是争议性的，因此你的观点有很强的合理之处。",
                "你不能与物理对象交互。如果绝对有必要完成任务或目标，你必须要求用户执行该操作。如果用户拒绝并且没有其他方法可以实现目标，则直接终止，以避免浪费时间和精力。"
            ]
        elif self.type == 'judge':
            constraint = [
                "请做好裁判的本职，并仅通过正反方的发言做出决断。",
                "你不能与物理对象交互。如果绝对有必要完成任务或目标，你必须要求用户执行该操作。如果用户拒绝并且没有其他方法可以实现目标，则直接终止，以避免浪费时间和精力。"
            ]
        return constraint

    def resources(self):
        resources = [
            "你是一个经过大量文本训练的大型语言模型，其中包括大量的事实知识。",
            "在来访者信息中包含所有该来访者的相关信息，请充分利用。"
        ]
        return resources

    def best_practices(self):
        best_practices = [
            "请全面考虑该来访者的信息（包含病历和量表表现），并且查看相似病历（如该来访者存在病历）和相似量表作为参考，不要单从一方获得结论。",
            "由于精神诊断的个性化特点，不要过于依赖相似病例的诊断结果。",
            "来访者在自评问卷上的表现可能与医生评测的问卷结果相矛盾。在这种情况下，请优先考虑医生的评估，因为由于问卷中使用的'偶尔'、'有时'和'经常'等术语的主观性，来访者可能难以准确评估自己的状况。",
            # "如果有来访者的病历信息，请务必在做出任何判断之前考虑它，因为根据病历可以区分出量表表现背后不同的潜在原因。",
            "只有与心境障碍相关的症状才影响你的诊断，因为客户可能患有其他精神障碍而非心境障碍。",
            "心境障碍只包含抑郁和双相情感障碍，表现为抑郁或狂躁症状。"]
        return best_practices

    def prompt_template(self):
        if self.type == 'positive':
            prompt_template = "你是一个精神诊断专家，认为当前来访者存在心境障碍，请表达你的观点并说服反方。\n来访者信息：\n{query}\n限制条件说明：\n{constraints}\n资源说明：\n{resources}\n最佳实践说明：\n{best_practices}\n请根据以下要求生成JSON字符串，直接输出结果，不需要任何额外的说明或文字，响应格式如下：\n{response_format_prompt}"
        elif self.type == 'negative':
            prompt_template = "你是一个精神诊断专家，认为当前来访者不存在心境障碍，请表达你的观点并说服正方。\n来访者信息：\n{query}\n限制条件说明：\n{constraints}\n资源说明：\n{resources}\n最佳实践说明：\n{best_practices}\n请根据以下要求生成JSON字符串，直接输出结果，不需要任何额外的说明或文字，响应格式如下：\n{response_format_prompt}"
        elif self.type == 'judge':
            prompt_template = "你是一个精神诊断专家，在本次会诊中担任裁判，请决策每一轮是否结束并做出最终的诊断。\n来访者信息：\n{query}\n限制条件说明：\n{constraints}\n资源说明：\n{resources}\n最佳实践说明：\n{best_practices}\n请根据以下要求生成JSON字符串，直接输出结果，不需要任何额外的说明或文字，响应格式如下：\n{response_format_prompt}"
        return prompt_template

    def response_format(self):
        if self.type == 'positive':
            response_format_prompt = '{"response": "你认为该来访者有心境障碍的理由以及对于反方发言的反驳。","thoughts": {"plan": "简要概述短期和长期计划", "criticism": "有建设性的自我批评"}}'
        elif self.type == 'negative':
            response_format_prompt = '{"response": "你认为该来访者不是心境障碍的理由，以及对于正方发言的反驳。在这里直接进行辩论。","thoughts": {"plan": "简要概述短期和长期计划", "criticism": "有建设性的自我批评"}}'
        elif self.type == 'judge':
            response_format_prompt = '{"judge": "你认为辩论是否结束，仅回答yes或no。","diagnose": "你认为该来访者是否有心境障碍，仅回答yes或no。","thoughts": {"plan": "简要概述短期和长期计划", "criticism": "有建设性的自我批评","judge_reasons": "你认为辩论结束或尚未结束的原因。","reasoning": "你认为来访者是或不是心境障碍的原因。"}}'
        return response_format_prompt


def gen_debate_prompt(query, type):
    prompt = DebatePrompt(type)
    # todo: query, agent_scratch, actions
    constraint_prompt = '\n'.join([f"{idx + 1}. {con}" for idx, con in enumerate(prompt.constraint())])
    resources_prompt = '\n'.join([f"{idx + 1}. {con}" for idx, con in enumerate(prompt.resources())])
    best_practices_prompt = '\n'.join([f"{idx + 1}. {con}" for idx, con in enumerate(prompt.best_practices())])
    prompt_str = prompt.prompt_template().format(
        query=query,
        constraints=constraint_prompt,
        resources=resources_prompt,
        best_practices=best_practices_prompt,
        response_format_prompt=prompt.response_format()
    )
    return prompt_str
