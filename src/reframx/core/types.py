"""Shared typing aliases used by the core package."""

from __future__ import annotations

from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str,"JSONValue"]
JSONObject: TypeAlias = dict[str,JSONValue]

ComponentRole: TypeAlias = str
ComponentName: TypeAlias = str
CapabilityName: TypeAlias = str
RequirementName: TypeAlias = str