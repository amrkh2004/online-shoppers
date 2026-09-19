"""
Pydantic Schemas for FastAPI Endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ShopperInputSchema(BaseModel):
    Administrative: int = Field(0, ge=0, description="Number of administrative pages visited")
    Administrative_Duration: float = Field(0.0, ge=0.0, description="Total time spent on administrative pages")
    Informational: int = Field(0, ge=0, description="Number of informational pages visited")
    Informational_Duration: float = Field(0.0, ge=0.0, description="Total time spent on informational pages")
    ProductRelated: int = Field(1, ge=0, description="Number of product-related pages visited")
    ProductRelated_Duration: float = Field(10.0, ge=0.0, description="Total time spent on product-related pages")
    BounceRates: float = Field(0.0, ge=0.0, le=1.0, description="Average bounce rate of pages visited")
    ExitRates: float = Field(0.02, ge=0.0, le=1.0, description="Average exit rate of pages visited")
    PageValues: float = Field(0.0, ge=0.0, description="Average page value of pages visited")
    SpecialDay: float = Field(0.0, ge=0.0, le=1.0, description="Closeness of site visiting time to a specific special day")
    Month: str = Field("May", description="Month of the visit (e.g. Feb, Mar, May, Oct, Nov)")
    OperatingSystems: int = Field(1, description="Operating system used by visitor")
    Browser: int = Field(2, description="Browser used by visitor")
    Region: int = Field(1, description="Geographic region of visitor")
    TrafficType: int = Field(1, description="Traffic source type")
    VisitorType: str = Field("Returning_Visitor", description="Visitor type (Returning_Visitor, New_Visitor, Other)")
    Weekend: bool = Field(False, description="Whether the visit date is on a weekend")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Administrative": 0,
                "Administrative_Duration": 0.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 12,
                "ProductRelated_Duration": 310.0,
                "BounceRates": 0.0,
                "ExitRates": 0.01,
                "PageValues": 38.5,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "Returning_Visitor",
                "Weekend": False,
            }
        }
    }


class BatchShopperInputSchema(BaseModel):
    inputs: List[ShopperInputSchema]


class PredictionResponseSchema(BaseModel):
    prediction: int = Field(..., description="0 for No Purchase, 1 for Purchase")
    label: str = Field(..., description="Human readable result: 'Purchase' or 'No Purchase'")
    probability: float = Field(..., description="Predicted probability of purchase")
    threshold_used: float = Field(..., description="Classification threshold used")


class BatchPredictionResponseSchema(BaseModel):
    predictions: List[PredictionResponseSchema]
    total_count: int


class HealthResponseSchema(BaseModel):
    status: str
    version: str


class MetadataResponseSchema(BaseModel):
    model_name: str
    version: str
    num_features: int
    classification_threshold: float
    baseline_metrics: dict
