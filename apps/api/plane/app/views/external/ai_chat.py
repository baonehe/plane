# Copyright (c) 2023-present Plane Software, Inc. and contributors
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

# Python imports
import os
from typing import Any, Dict, List

# Django imports
from django.utils import timezone

# Third party imports
import openai
import pytz
from openai import OpenAI
from rest_framework import status
from rest_framework.response import Response

# Module imports
from plane.app.permissions import ROLE, allow_permission
from plane.db.models import Workspace
from plane.utils.exception_logger import log_exception
from plane.utils.host import base_host

from ..base import BaseAPIView
from .ai_chat_tools import TOOL_DEFINITIONS, WorkItemTools
from .base import get_llm_config

MAX_HISTORY_MESSAGES = 20
MAX_MESSAGE_LENGTH = 4000
MAX_TOOL_ROUNDS = 8
LLM_TIMEOUT_SECONDS = 60
CHAT_ROLES = ("user", "assistant")
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "medium")


class ChatValidationError(Exception):
    pass


def parse_chat_history(messages: Any) -> List[Dict[str, str]]:
    """Validate the client-held history and keep only the latest turns."""
    if not isinstance(messages, list) or not messages:
        raise ChatValidationError("messages must be a non-empty list")

    history = []
    for message in messages[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(message, dict):
            raise ChatValidationError("Each message must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in CHAT_ROLES or not isinstance(content, str):
            raise ChatValidationError("Each message needs a role of user/assistant and a string content")
        history.append({"role": role, "content": content[:MAX_MESSAGE_LENGTH]})

    if history[-1]["role"] != "user" or not history[-1]["content"].strip():
        raise ChatValidationError("The last message must be a non-empty user message")
    return history


def build_system_prompt(user, workspace: Workspace) -> str:
    try:
        user_timezone = pytz.timezone(user.user_timezone or "UTC")
    except pytz.UnknownTimeZoneError:
        user_timezone = pytz.UTC
    today = timezone.now().astimezone(user_timezone).strftime("%A, %Y-%m-%d")
    return (
        "You are Plane AI, an assistant inside the Plane project management app. "
        f"You help {user.display_name} manage work items (tasks) in the workspace '{workspace.name}'. "
        f"Today is {today} ({user_timezone.zone}).\n"
        "Rules:\n"
        "- Always use the tools to read or change data; never invent IDs, references or results.\n"
        "- Resolve project, state, label and member names to IDs with list_projects and get_project_details "
        "before creating or updating work items.\n"
        "- If a request is ambiguous (several matching projects, work items or people), ask a short "
        "clarifying question instead of guessing.\n"
        "- Refer to work items by their reference, e.g. WEB-12.\n"
        "- After changing data, briefly confirm what changed.\n"
        "- You cannot delete work items; tell the user to do that in the app.\n"
        "- Reply in the same language the user writes in. Keep answers concise; use plain text with "
        "simple '-' bullet lists, no markdown tables or headings."
    )


def create_response(client: OpenAI, instructions: str, input_items: List[Any], options: Dict[str, Any]):
    """Request a model response with the configured reasoning effort.

    Models without reasoning (e.g. gpt-4o-mini) reject the ``reasoning`` options; in that case
    they are dropped from ``options`` so the retry and later tool rounds of the same chat turn omit them.
    """
    try:
        return client.responses.create(instructions=instructions, input=input_items, **options)
    except openai.BadRequestError as e:
        if "reasoning" not in options or "reasoning" not in str(e):
            raise
        options.pop("reasoning")
        options.pop("include", None)
        return client.responses.create(instructions=instructions, input=input_items, **options)


class WorkspaceAIChatEndpoint(BaseAPIView):
    @allow_permission(allowed_roles=[ROLE.ADMIN, ROLE.MEMBER], level="WORKSPACE")
    def post(self, request, slug):
        try:
            history = parse_chat_history(request.data.get("messages"))
        except ChatValidationError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        api_key, model, provider = get_llm_config()
        if not api_key or not model or not provider:
            return Response(
                {"error": "LLM provider API key and model are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if provider.lower() != "openai":
            return Response(
                {"error": "AI chat currently supports only the OpenAI provider"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        workspace = Workspace.objects.get(slug=slug)
        tools = WorkItemTools(request.user, workspace, origin=base_host(request=request, is_app=True))
        instructions = build_system_prompt(request.user, workspace)
        input_items: List[Any] = list(history)

        client = OpenAI(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)
        options: Dict[str, Any] = {
            "model": model,
            "tools": TOOL_DEFINITIONS,
            "reasoning": {"effort": LLM_REASONING_EFFORT},
            # Nothing is stored at OpenAI, so reasoning is passed back encrypted between tool rounds
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                response = create_response(client, instructions, input_items, options)
                calls = [item for item in response.output if item.type == "function_call"]
                if not calls:
                    return Response(
                        {"message": response.output_text or "", "actions": tools.actions},
                        status=status.HTTP_200_OK,
                    )
                # The model needs its own reasoning and function call items alongside their outputs
                input_items.extend(response.output)
                for call in calls:
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": tools.execute(call.name, call.arguments),
                        }
                    )
        except Exception as e:
            log_exception(e)
            # Actions already applied are still returned so the client can refresh them
            return Response(
                {"error": "Error occurred while generating a response from the AI provider", "actions": tools.actions},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        return Response(
            {
                "message": "This request needed too many steps. Please split it into smaller requests.",
                "actions": tools.actions,
            },
            status=status.HTTP_200_OK,
        )
