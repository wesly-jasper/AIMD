from pydantic import BaseModel,Field
from typing import Any


class DetectionRegion(BaseModel):
    type:str
    confidence:float=Field(ge=0.0,le=1.0)
    bbox:list[float]|None=None
    mask_path:str|None=None
    start_frame:int|None=None
    end_frame:int|None=None
    start_timestamp:float|None=None
    end_timestamp:float|None=None


class DetectionResult(BaseModel):
    detector:str
    media_type:str
    manipulation_detected:bool
    confidence:float=Field(ge=0.0,le=1.0)
    manipulation_type:str
    regions:list[DetectionRegion]=Field(default_factory=list)
    metadata:dict[str,Any]=Field(default_factory=dict)

class DetectionRequest(BaseModel):
    file_path:str
    media_type:str