"""Schemas Pydantic v2 (validation stricte)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["yes", "no"]
Freq = Literal["no", "Sometimes", "Frequently", "Always"]


class LifestyleFeatures(BaseModel):
    """Variables du pipeline A (sans Height ni Weight)."""

    model_config = ConfigDict(extra="forbid")

    Gender: Literal["Female", "Male"]
    Age: float = Field(ge=14, le=65)
    family_history_with_overweight: YesNo
    FAVC: YesNo
    FCVC: int = Field(ge=1, le=3, description="Frequence de consommation de legumes")
    NCP: int = Field(ge=1, le=4, description="Nombre de repas principaux")
    CAEC: Freq
    SMOKE: YesNo
    CH2O: int = Field(ge=1, le=3, description="Consommation d'eau")
    SCC: YesNo
    FAF: int = Field(ge=0, le=3, description="Activite physique")
    TUE: int = Field(ge=0, le=2, description="Temps d'ecran")
    CALC: Freq
    MTRANS: Literal["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"]


class WeightFeatures(LifestyleFeatures):
    """Variables du pipeline B (avec Height)."""

    Height: float = Field(ge=1.3, le=2.1, description="Taille en metres")


class ClassPrediction(BaseModel):
    label: str
    probabilities: dict[str, float] | None = None


class WeightPrediction(BaseModel):
    weight_kg: float
    bmi: float
    bmi_class: str


class Health(BaseModel):
    status: str
    classifier: str | None
    regressor: str | None
