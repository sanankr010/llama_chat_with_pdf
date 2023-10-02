#for webui class
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests
import json


class webuiLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"


    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = requests.post(
            "http://34.139.176.110:8080/v1/models/model:predict",   #34.139.90.39 A100 #34.42.166.84 16gb
            json={
                'prompt': prompt
            }
        )
        print(response)
        response.raise_for_status()
        print(response.json())
        return response.json()['data'][0]["generated_text"].strip().replace("```", " ")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {

        }
