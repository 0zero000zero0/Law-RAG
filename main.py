import json
import os
from pathlib import Path
from pprint import pp

from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from data import get_test_data_from_pkl
from db import get_db, get_similar_cases
from feature import extract_feature_from_fact, extract_fields_from_feature
from prompt import get_prompt_template_from_file
from related_laws import get_related_laws

# !note: your api
api_key = os.getenv("API_KEY", "")
base_api = "https://api.siliconflow.cn/v1"
llm = ChatOpenAI(
    model="Qwen/Qwen3-32B",
    base_url=base_api,
    api_key=api_key,  # type: ignore
    temperature=0.7,
    top_p=0.9,
)

embedding_model = SentenceTransformer("BAAI/bge-m3")
collection_name = "legal_cases"
client = get_db(db_path="db/case.db", collection_name=collection_name)
prompts_dir = Path("prompts")
feature_prompt = get_prompt_template_from_file(prompts_dir / "extract_feature.txt", input_variables=["fact"])
infer_prompt = get_prompt_template_from_file(
    prompts_dir / "infer.txt", input_variables=["fact", "feature", "related_laws", "similar_cases"]
)


test_data_path = Path("pkl") / "sampled_data.pkl"
test_data = get_test_data_from_pkl(test_data_path)
fact = test_data[0]["fact"]

print(f"正在从案情描述中抽取特征")
feature = extract_feature_from_fact(fact, feature_prompt, llm)
pp(f"特征抽取完成:\n{feature}")

print(f"正在查询相关法律条文")
laws_results = get_related_laws(fact, topk=3)
pp(f"相关法律条文:\n{laws_results}")

print(f"正在查询类案")
field_dict = extract_fields_from_feature(feature)
similar_cases = get_similar_cases(
    field_dict=field_dict,
    embedding_model=embedding_model,
    client=client,
    collection_name=collection_name,
    search_limit=9,
    return_topk=3,
)
pp(f"相似案例:\n{similar_cases}")

print(f"正在进行最终判决预测")
infer_prompt_format = infer_prompt.format(
    fact=fact,
    feature=feature,
    related_laws=laws_results,
    similar_cases=similar_cases,
)
response = llm.invoke(infer_prompt_format)
result = json.loads(response.content)  # type: ignore
print(f"判决结果:\n{result}")

client.close()
