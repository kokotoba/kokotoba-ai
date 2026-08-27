"""発話カード生成・選択API。"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.card_suggestion_service import (
    CardNotFoundError,
    CardSuggestionService,
    SelectionConflictError,
    SuggestionNotFoundError,
)
from database.card_suggestion_repository import CardSuggestion

router = APIRouter(prefix="/v1/card-suggestions", tags=["card-suggestions"])


class SuggestionContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    place_name: str = Field(min_length=1, max_length=200)
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)

    @field_validator("place_name")
    @classmethod
    def strip_place_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("place_name must not be blank")
        return value


class CreateSuggestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    context: SuggestionContextRequest
    mode: Literal["fast", "quality"] = "fast"

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class CardResponse(BaseModel):
    id: str
    text: str


class SuggestionResponse(BaseModel):
    id: str
    question_type: Literal["yes_no", "choice", "open"]
    mode: Literal["fast", "quality"]
    cards: list[CardResponse]
    created_at: str


class SelectCardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_id: str = Field(min_length=1, max_length=100)

    @field_validator("card_id")
    @classmethod
    def strip_card_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("card_id must not be blank")
        return value


class SelectionResponse(BaseModel):
    suggestion_id: str
    selected_card: CardResponse
    recorded_at: str


def get_card_suggestion_service(request: Request) -> CardSuggestionService:
    return request.app.state.card_suggestion_service


ServiceDependency = Annotated[
    CardSuggestionService,
    Depends(get_card_suggestion_service),
]


def _to_response(suggestion: CardSuggestion) -> SuggestionResponse:
    return SuggestionResponse(
        id=suggestion.id,
        question_type=suggestion.question_type,
        mode=suggestion.generation_mode,
        cards=[
            CardResponse(id=card.id, text=card.text)
            for card in suggestion.cards
        ],
        created_at=suggestion.created_at,
    )


@router.post(
    "",
    response_model=SuggestionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_suggestion(
    payload: CreateSuggestionRequest,
    service: ServiceDependency,
) -> SuggestionResponse:
    suggestion = service.create_suggestion(
        question=payload.question,
        location=payload.context.place_name,
        latitude=payload.context.latitude,
        longitude=payload.context.longitude,
        fast=payload.mode == "fast",
    )
    return _to_response(suggestion)


@router.post(
    "/{suggestion_id}/selection",
    response_model=SelectionResponse,
)
def select_card(
    suggestion_id: str,
    payload: SelectCardRequest,
    service: ServiceDependency,
) -> SelectionResponse:
    try:
        _, selected_card, selected_at = service.select_card(
            suggestion_id,
            payload.card_id,
        )
    except SuggestionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Card suggestion was not found.",
        ) from exc
    except CardNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="card_id is not included in this suggestion.",
        ) from exc
    except SelectionConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A different card has already been selected.",
        ) from exc

    return SelectionResponse(
        suggestion_id=suggestion_id,
        selected_card=CardResponse(
            id=selected_card.id,
            text=selected_card.text,
        ),
        recorded_at=selected_at,
    )
