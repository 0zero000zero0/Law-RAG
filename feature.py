import json
from typing import Any


def format_feature_output(generated_text: str):
    feature = generated_text.split("```")[-2]
    feature = feature.replace("json", "")
    feature = feature.replace("\xa0", "")
    feature = json.loads(feature)
    return feature


def extract_feature_from_fact(fact: str, feature_prompt, llm):
    """从生成的文本中提取特征"""
    feature_prompt_format = feature_prompt.format(
        fact=fact,
    )
    response = llm.invoke(feature_prompt_format)
    content: str = response.content
    feature = format_feature_output(content)
    return feature


def extract_fields_from_feature(case: dict[str, Any]) -> dict:  # noqa: D103
    result = {}
    for k, v in case.items():
        desciption = ""
        if isinstance(v, dict):
            for i, j in v.items():
                if isinstance(j, list):
                    if isinstance(j, str):
                        desciption += ", ".join(j)
                    elif isinstance(j, dict):
                        desciption += json.dumps(j, ensure_ascii=False)
                elif isinstance(j, dict):
                    for a, b in j.items():
                        desciption += f"{i}: {a}: {b}\n"
                else:
                    desciption += f"{i}: {j}\n"
        elif isinstance(v, list):
            desciption += ", ".join(str(x) for x in v) + "\n"
        else:
            desciption += f"{v}\n"
        result[k] = desciption
    return result
