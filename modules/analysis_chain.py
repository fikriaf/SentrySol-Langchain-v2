import os
from datetime import datetime
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_cerebras import ChatCerebras
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "qwen-3-coder-480b")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-medium")

CEREBRAS_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

prompt_template = PromptTemplate(
    input_variables=["context"],
    template="""You are a blockchain threat intelligence analyst specializing in detecting malicious wallet activities.

Analyze the following combined JSON data from SentrySol Security & SentrySol Blockchain Analyzer APIs.

TASKS:
1. Identify all potential threats (e.g., phishing, scam, dusting, spoofing, approval exploits, rug pulls, laundering patterns).
2. For each threat:
   - threat_type
   - reason (detailed and specific)
   - confidence (Low, Medium, High)
   - supporting_evidence
   - recommended_actions
3. Provide overall_risk_level ( **minimal, low, medium, high, critical** ), risk_score ( **scale 100** ), risk_factors, ioc, and additional_notes.

Respond in valid JSON only. Do not mention "Metasleuth" and "Helius" in your response.

FORMAT OUTPUT:
{{
  "threat_analysis": {{
    "metadata": {{
      "target_address": "...",
      "chain": "Solana",
      "analysis_timestamp": "{timestamp}",
      "data_sources": ["SentrySol Security", "SentrySol Blockchain Analyzer", "SentrySol ML Model"]
    }},
    "potential_threats": [...],
    "overall_risk_level": "...",
    "risk_score": ...,
    "risk_factors": [...],
    "ioc": {{
      "addresses": [...],
      "transaction_signatures": [...],
      "suspicious_mints": [...],
      "related_programs": [...]
    }},
    "additional_notes": "...",
    "engine": "{engine}"
  }}
}}

THIS IS THE DATA:
==================================================================
{context}
==================================================================
""",
)


def run_analysis(context: str):
    local_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        if not CEREBRAS_API_KEY:
            raise ValueError("CEREBRAS_API_KEY (OPENAI_API_KEY) not set")
        llm = ChatCerebras(model=CEREBRAS_MODEL, openai_api_key=CEREBRAS_API_KEY, temperature=0.7, max_tokens=4000)
        chain = LLMChain(llm=llm, prompt=prompt_template)
        return chain.run(context=context, timestamp=local_timestamp, engine="SentrySol-Premium SDK")
    except Exception as cerebras_exc:
        print(f"Error with Cerebras: {str(cerebras_exc)}")
        try:
            llm = ChatMistralAI(
                model=LLM_MODEL, mistral_api_key=MISTRAL_API_KEY, temperature=0
            )
            chain = LLMChain(llm=llm, prompt=prompt_template)
            return chain.run(context=context, timestamp=local_timestamp, engine="SentrySol SDK")
        except Exception as mistral_exc:
            print(f"Error with Mistral AI: {str(mistral_exc)}")
