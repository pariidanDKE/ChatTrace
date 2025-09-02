from typing_extensions import List, TypedDict, Annotated
from typing import Optional, Literal

class WhatsAppSearch(TypedDict):
    """Information about a query, speifically particpants and dates."""
    participants: Annotated[
        Optional[List[str]],
        ...,
        "If there are no names of a specific person in the query leave the field, empty."
        "Name of person if mentioned in the question."
    ]
    
    time_period: Annotated[
        Optional[Literal["day", "month", "year"]],
        ...,
        "Type of time period: day, month, year, leave empty if no time period mentioned."
    ]
    day: Annotated[Optional[int], ..., "Day number (1-31). Only set for day-specific queries."]
    month: Annotated[Optional[int], ..., "Month number (1-12). Set for day and month queries."]
    year: Annotated[Optional[int], ..., "Year number (e.g. 2023). Set for all time queries."]
    time_query: Annotated[
        Optional[Literal["within", "before", "after"]],
        ...,
        "Time relationship: 'within' = DURING (e.g. 'conversations 2024', 'in March', 'from 2023', 'during June'), 'before' = PRIOR TO (e.g. 'before 2024', 'prior to March'), 'after' = FOLLOWING (e.g. 'after 2024', 'since March'). Leave empty if no time relationship mentioned."
    ]

def analyze_query(chat_model, question: str) -> WhatsAppSearch:
    """Analyze user question with separate date fields."""
    structured_llm = chat_model.with_structured_output(schema=WhatsAppSearch)
    analysis_prompt = (
        "If present, Extract from this WhatsApp question:\n"
        "- Person names mentioned\n"
        "- Time references and whether it's before/after/within a time period\n"
        "- Extract date components separately:\n"
        " * 'March 2023' → month: 3, year: 2023\n"
        " * '2023' → year: 2023\n"
        " * 'June 15, 2024' → day: 15, month: 6, year: 2024\n"
        " Do NOT hallucinate names or dates if they are not there.\n"
        f"Question: {question}"
    )
    query_analysis = structured_llm.invoke(analysis_prompt)
    query_analysis['query'] = question
    return query_analysis