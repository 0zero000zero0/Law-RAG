import requests


def get_related_laws(
    fact: str,
    topk=5,
    collection_name="article",
    url="http://101.132.61.37:5007/search",
    headers={"Authorization": "Bearer mysecrettoken"},
):
    data = {"collection_name": collection_name, "query": fact, "topk": topk}
    try:
        response = requests.post(url, json=data, headers=headers).json()
        if len(response.get("data")) == 0:
            raise ValueError("相关法律条文为空")
        related_laws = {"related_laws": []}
        for item in response["data"]:
            law = {
                "title": item["metadata"]["title"],
                "content": item["metadata"]["content"],
                "score": item["score"],
            }
            related_laws["related_laws"].append(law)
        return related_laws
    except Exception as e:
        return None, {"error": str(e), "response": response}
