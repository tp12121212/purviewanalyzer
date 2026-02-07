from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("entity_key", name="uq_entities_entity_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_key: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    entity_type: Mapped[str] = mapped_column(String, nullable=False)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    recognizer_type: Mapped[str] = mapped_column(String, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    source_file: Mapped[str] = mapped_column(String, nullable=False)
    source_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    patterns: Mapped[list[EntityPattern]] = relationship(
        back_populates="entity", cascade="all, delete-orphan"
    )
    contexts: Mapped[list[EntityContext]] = relationship(
        back_populates="entity", cascade="all, delete-orphan"
    )
    metadata_items: Mapped[list[EntityMetadata]] = relationship(
        back_populates="entity", cascade="all, delete-orphan"
    )


class EntityPattern(Base):
    __tablename__ = "entity_patterns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), nullable=False)
    pattern_name: Mapped[str | None] = mapped_column(String, nullable=True)
    regex: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    entity: Mapped[Entity] = relationship(back_populates="patterns")


class EntityContext(Base):
    __tablename__ = "entity_context"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), nullable=False)
    context: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    entity: Mapped[Entity] = relationship(back_populates="contexts")


class EntityMetadata(Base):
    __tablename__ = "entity_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), nullable=False)
    key: Mapped[str] = mapped_column(String, nullable=False)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    entity: Mapped[Entity] = relationship(back_populates="metadata_items")


class Recognizer(Base):
    __tablename__ = "recognizers"
    __table_args__ = (
        UniqueConstraint(
            "name", "entity_type", "language", name="uq_recognizers_name_entity_lang"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    entity_type: Mapped[str] = mapped_column(String, nullable=False)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    base_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    allow_list_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    deny_list_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_root: Mapped[str] = mapped_column(String, nullable=False)
    storage_subpath: Mapped[str | None] = mapped_column(String, nullable=True)
    module_path: Mapped[str] = mapped_column(String, nullable=False)
    class_name: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    patterns: Mapped[list[RecognizerPattern]] = relationship(
        back_populates="recognizer", cascade="all, delete-orphan"
    )
    contexts: Mapped[list[RecognizerContext]] = relationship(
        back_populates="recognizer", cascade="all, delete-orphan"
    )


class RecognizerPattern(Base):
    __tablename__ = "recognizer_patterns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recognizer_id: Mapped[int] = mapped_column(
        ForeignKey("recognizers.id"), nullable=False
    )
    pattern_name: Mapped[str | None] = mapped_column(String, nullable=True)
    regex: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    recognizer: Mapped[Recognizer] = relationship(back_populates="patterns")


class RecognizerContext(Base):
    __tablename__ = "recognizer_context"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recognizer_id: Mapped[int] = mapped_column(
        ForeignKey("recognizers.id"), nullable=False
    )
    context: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    recognizer: Mapped[Recognizer] = relationship(back_populates="contexts")
