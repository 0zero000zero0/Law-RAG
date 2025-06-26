from pathlib import Path
from langchain_core.prompts import PromptTemplate


def get_prompt_template_from_file(prompt_file_path, input_variables: list[str]):
    with open(prompt_file_path) as f:
        template = f.read()
    prompt_template = PromptTemplate(
        template=template,
        input_variables=input_variables,
    )
    return prompt_template
