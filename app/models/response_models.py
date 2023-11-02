from pydantic import BaseModel


class AgentResponse(BaseModel):
    answer: str
    status: str

class AskQuestionResponse(BaseModel):
    answer: str
    status: str