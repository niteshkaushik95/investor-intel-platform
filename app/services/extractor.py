import json
import logging
from groq import Groq
from app.schemas import ExtractionResponse
from app import config

logger = logging.getLogger("ExtractorService")

# Centralized API key management
client = Groq(api_key=config.GROQ_API_KEY)

def extract_financial_data(markdown_text: str) -> ExtractionResponse:
    """
    Feeds the full document text to the LLM and forces it to extract 
    structured financial metrics and qualitative insights.
    """
    
    system_prompt = """
    You are an elite FinTech AI data extractor. Your job is to read financial documents 
    and extract exact numerical metrics and key qualitative insights.
    
    You MUST output valid JSON matching this exact structure:
    {
      "metrics": [
        {
          "metric_name": "Revenue",
          "numerical_value": "15000000.0",
          "currency": "USD",
          "unit": "absolute",
          "year": 2026,
          "period_enum": "OTHER",
          "period_description": "Nine months ended September 30, 2026",
          "page_number": 12,
          "section_name": "Financial Results",
          "source_snippet": "Revenue for the nine months ended September 30, 2026 was $15.0 million."
        }
      ],
      "insights": [
        {
          "insight_name": "Primary Risk Factor",
          "content_text": "Supply chain disruptions are compressing margins.",
          "page_number": 15,
          "section_name": "Risk Factors",
          "source_snippet": "We expect continued margin pressure from overseas supply chain delays."
        }
      ]
    }
    
    Rules:
    1. Do not hallucinate numbers. If it is not in the text, do not extract it.
    2. The source_snippet MUST be an exact verbatim quote from the text. Do not summarize it.
    3. Convert textual numbers (e.g., "15 million") into their exact normalized value (e.g., "15000000.0"). Return numerical_value as a decimal-compatible string without currency symbols or commas.
    4. For percentages, extract the absolute number (e.g., "42.5") and set the unit to "%".
    """

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract data from this text:\n\n{markdown_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0, 
        )
        
        raw_json_string = response.choices[0].message.content
        parsed_json = json.loads(raw_json_string)
        
        validated_data = ExtractionResponse(**parsed_json)
        return validated_data

    except Exception as e:
        logger.error(f"LLM Extraction failed: {str(e)}")
        raise e