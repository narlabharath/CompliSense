import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnableSequence

# Initialize the LLM
temperature = 0
llm = ChatOpenAI(temperature=temperature)

# ========== ENTITY SCHEMA FOR OUTPUT PARSER ==========
class ProductEntity(BaseModel):
    product_name: str = Field(..., description="Name of the product")
    product_type: str = Field(..., description="Type of product")
    order_validity: str = Field(..., description="Order validity period")
    quantity_or_amount: str = Field(..., description="Quantity or amount involved")
    price: str = Field(..., description="Price of the product")
    ccy: str = Field(..., description="Currency")
    product_features: str = Field(..., description="General product features discussed")
    timestamp: str = Field(..., description="Timestamp when product was discussed")

# ========== 1. ENTITY EXTRACTION ==========
def extract_entities(transcript: str, product_types: List[str]) -> List[Dict[str, Any]]:
    parser = StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="product_name", description="Name of the product"),
        ResponseSchema(name="product_type", description="Type of product (must match provided list)"),
        ResponseSchema(name="order_validity", description="Order validity period"),
        ResponseSchema(name="quantity_or_amount", description="Quantity or amount involved"),
        ResponseSchema(name="price", description="Price of the product"),
        ResponseSchema(name="ccy", description="Currency used"),
        ResponseSchema(name="product_features", description="Product features discussed"),
        ResponseSchema(name="timestamp", description="Timestamp when product was discussed")
    ])

    format_instructions = parser.get_format_instructions()

    template = """
    You are a financial assistant. Extract all products discussed in the transcript. 
    For each product, extract the following:
    - product_name
    - product_type (must be one of: {product_types})
    - order_validity
    - quantity_or_amount
    - price
    - ccy (currency)
    - product_features
    - timestamp

    Output must be in the following format:
    {format_instructions}

    Transcript:
    {transcript}
    """

    prompt = PromptTemplate(
        input_variables=["transcript", "product_types"],
        template=template,
        partial_variables={"format_instructions": format_instructions},
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    raw_output = chain.run(transcript=transcript, product_types=", ".join(product_types))

    try:
        parsed_output = parser.parse(raw_output)
        return [parsed_output] if isinstance(parsed_output, dict) else parsed_output
    except Exception as e:
        return [{"error": str(e), "raw_output": raw_output}]


# ========== 2. ORDER TAKING CHECKLIST ==========
def check_order_taking(transcript: str, product_type: str, checklist_folder: str = "products_checklist") -> Dict[str, Any]:
    checklist_path = os.path.join(checklist_folder, f"{product_type}.txt")
    if not os.path.exists(checklist_path):
        return {"status": "error", "message": f"Checklist not found for {product_type}"}

    with open(checklist_path, 'r') as f:
        checklist = f.read()

    template = """
    You are a compliance assistant. Check whether the RM in the below transcript has followed all the points in the checklist for the product type: {product_type}.

    Checklist:
    {checklist}

    Transcript:
    {transcript}

    Provide a summary with:
    - Whether the checklist was followed
    - Pass/Fail flag
    - Reasons for failure (if any)
    """
    prompt = PromptTemplate(
        input_variables=["transcript", "product_type", "checklist"],
        template=template,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(transcript=transcript, product_type=product_type, checklist=checklist)
    return {"product_type": product_type, "check_result": result}


# ========== 3. SALES SUITABILITY CHECKLIST ==========
def check_sales_suitability(transcript: str, entities: List[Dict[str, Any]], risk_scenarios: Dict[str, str]) -> List[Dict[str, Any]]:
    results = []
    for entity in entities:
        product_type = entity.get("product_type")
        scenario_script = risk_scenarios.get(product_type)
        if not scenario_script:
            continue

        template = """
        Check whether the following risk scenario script was properly communicated by the RM in the transcript.
        
        Risk Scenario for {product_type}:
        {scenario_script}

        Transcript:
        {transcript}

        Result:
        - Did the RM disclose this risk scenario?
        - Yes/No
        - Justification
        """
        prompt = PromptTemplate(
            input_variables=["transcript", "product_type", "scenario_script"],
            template=template,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(transcript=transcript, product_type=product_type, scenario_script=scenario_script)
        results.append({"product_type": product_type, "suitability_result": result})

    return results


# ========== 4. LANGGRAPH EXECUTION SEQUENCE ==========
def run_compliance_pipeline(transcript: str, product_types: List[str], risk_scenarios: Dict[str, str]) -> Dict[str, Any]:
    def extract(transcript_and_types):
        transcript, product_types = transcript_and_types["transcript"], transcript_and_types["product_types"]
        return {"entities": extract_entities(transcript, product_types)}

    def check_orders(data):
        transcript = data["transcript"]
        entities = data["entities"]
        order_checks = [check_order_taking(transcript, entity["product_type"]) for entity in entities if "product_type" in entity]
        return {"order_checks": order_checks, **data}

    def check_suitability(data):
        return {"suitability_checks": check_sales_suitability(data["transcript"], data["entities"], data["risk_scenarios"]), **data}

    pipeline = RunnableSequence(
        steps=[
            RunnableLambda(extract),
            RunnableLambda(check_orders),
            RunnableLambda(check_suitability),
        ]
    )

    inputs = {"transcript": transcript, "product_types": product_types, "risk_scenarios": risk_scenarios}
    return pipeline.invoke(inputs)
