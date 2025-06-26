import json
from collections import defaultdict

from pymilvus import MilvusClient


def get_db(db_path, collection_name="legal_cases"):
    client = MilvusClient(db_path)
    client.load_collection(collection_name)
    return client


def search_field(field_name, query, embedding_model, client, collection_name, limit=5):
    """对指定字段进行批量搜索"""
    query_vector = embedding_model.encode(query, convert_to_numpy=True)
    results = client.search(
        collection_name=collection_name,
        data=[query_vector.tolist()],
        anns_field=field_name,
        limit=limit,
        output_fields=["case_json"],
    )
    return results


def get_similar_cases(field_dict, embedding_model, client, collection_name, search_limit=9, return_topk=9):

    fields_to_search = [
        "crime_category",
        "criminal_composition",
        "evidence_characteristics",
    ]

    field_queries = {
        "crime_category": field_dict["罪名类别"],
        "criminal_composition": field_dict["犯罪构成"],
        "evidence_characteristics": field_dict["证据特征"],
    }

    field_results = {}
    for field in fields_to_search:
        field_results[field] = search_field(
            field, field_queries[field], embedding_model, client, collection_name, limit=search_limit
        )

    combined_scores = defaultdict(list)
    for field in fields_to_search:
        result_list = field_results[field]
        for hit in result_list[0]:
            case_id = hit["id"]
            distance = hit["distance"]
            case_json = hit["entity"].get("case_json")
            combined_scores[case_id].append({"distance": distance, "case_json": case_json})

        averaged_results = []
        for case_id, entries in combined_scores.items():
            distances = [e["distance"] for e in entries]
            avg_distance = sum(distances) / len(distances)
            sample_case = json.loads(entries[0]["case_json"])  # 提取 case_json
            averaged_results.append({"id": case_id, "avg_similarity": avg_distance, "case": sample_case})
        averaged_results.sort(key=lambda x: x["avg_similarity"])
        top_k_cases = averaged_results[:return_topk]

    return top_k_cases
