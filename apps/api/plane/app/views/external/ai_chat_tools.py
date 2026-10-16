# Copyright (c) 2023-present Plane Software, Inc. and contributors
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

"""Work-item tools exposed to the LLM behind the AI chat endpoint.

Every tool runs as the requesting user and is scoped to the projects of the
current workspace where that user is an active admin or member, mirroring the
permissions of the regular issue/comment endpoints.
"""

# Python imports
import json
import re
import uuid
from html import escape
from typing import Any, Callable, Dict, List, Optional

# Django imports
from django.contrib.postgres.aggregates import ArrayAgg
from django.contrib.postgres.fields import ArrayField
from django.core.exceptions import ValidationError as DjangoValidationError
from django.core.serializers.json import DjangoJSONEncoder
from django.db.models import Q, UUIDField, Value
from django.db.models.functions import Coalesce
from django.utils import timezone

# Module imports
from plane.app.permissions import ROLE
from plane.app.serializers import (
    IssueCommentSerializer,
    IssueCreateSerializer,
    IssueDetailSerializer,
)
from plane.bgtasks.issue_activities_task import issue_activity
from plane.bgtasks.issue_description_version_task import issue_description_version_task
from plane.bgtasks.webhook_task import model_activity
from plane.db.models import (
    Issue,
    IssueAssignee,
    IssueLabel,
    Label,
    Project,
    ProjectMember,
    State,
    Workspace,
)
from plane.utils.exception_logger import log_exception

WORK_ITEM_REF_PATTERN = re.compile(r"^([A-Za-z0-9]+)-(\d+)$")
PRIORITIES = [choice for choice, _ in Issue.PRIORITY_CHOICES]
STATE_GROUPS = ["backlog", "unstarted", "started", "completed", "cancelled"]
MAX_SEARCH_LIMIT = 50

_WORK_ITEM_FIELDS = {
    "name": {"type": "string", "description": "Title of the work item."},
    "description": {
        "type": "string",
        "description": "Plain-text description. Replaces the existing description.",
    },
    "priority": {"type": "string", "enum": PRIORITIES},
    "state_id": {
        "type": "string",
        "description": "State UUID from get_project_details.",
    },
    "assignee_ids": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Full list of assignee user UUIDs; replaces the current assignees.",
    },
    "label_ids": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Full list of label UUIDs; replaces the current labels.",
    },
    "start_date": {
        "type": ["string", "null"],
        "description": "YYYY-MM-DD, or null to clear.",
    },
    "target_date": {
        "type": ["string", "null"],
        "description": "Due date as YYYY-MM-DD, or null to clear.",
    },
    "parent": {
        "type": ["string", "null"],
        "description": "Parent work item reference (e.g. WEB-12), or null to clear.",
    },
}

_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "list_projects",
        "description": "List the projects in this workspace the user can manage work items in.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "get_project_details",
        "description": "Get the states, labels and members of a project. Use it to resolve names to IDs.",
        "parameters": {
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
        },
    },
    {
        "type": "function",
        "name": "search_work_items",
        "description": "Search work items across the user's projects, most recently updated first.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to match in the title, or a reference like WEB-12.",
                },
                "project_id": {"type": "string"},
                "assignee_id": {
                    "type": "string",
                    "description": "User UUID, or 'me' for the current user.",
                },
                "state_groups": {
                    "type": "array",
                    "items": {"type": "string", "enum": STATE_GROUPS},
                },
                "priorities": {
                    "type": "array",
                    "items": {"type": "string", "enum": PRIORITIES},
                },
                "due_before": {
                    "type": "string",
                    "description": "Only items with a due date on or before YYYY-MM-DD.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_LIMIT},
            },
        },
    },
    {
        "type": "function",
        "name": "get_work_item",
        "description": "Get the full details of one work item, including its description.",
        "parameters": {
            "type": "object",
            "properties": {
                "work_item": {
                    "type": "string",
                    "description": "Reference like WEB-12, or the work item UUID.",
                }
            },
            "required": ["work_item"],
        },
    },
    {
        "type": "function",
        "name": "create_work_item",
        "description": "Create a work item in a project.",
        "parameters": {
            "type": "object",
            "properties": {"project_id": {"type": "string"}, **_WORK_ITEM_FIELDS},
            "required": ["project_id", "name"],
        },
    },
    {
        "type": "function",
        "name": "update_work_item",
        "description": "Update fields of an existing work item. Only pass the fields that change.",
        "parameters": {
            "type": "object",
            "properties": {
                "work_item": {
                    "type": "string",
                    "description": "Reference like WEB-12, or the work item UUID.",
                },
                **_WORK_ITEM_FIELDS,
            },
            "required": ["work_item"],
        },
    },
    {
        "type": "function",
        "name": "add_comment",
        "description": "Add a comment to a work item.",
        "parameters": {
            "type": "object",
            "properties": {
                "work_item": {
                    "type": "string",
                    "description": "Reference like WEB-12, or the work item UUID.",
                },
                "comment": {"type": "string", "description": "Plain-text comment."},
            },
            "required": ["work_item", "comment"],
        },
    },
]

# Responses API function tools; strict schemas (its default) would make every argument required
TOOL_DEFINITIONS: List[Dict[str, Any]] = [{**tool, "strict": False} for tool in _TOOLS]


class ToolError(Exception):
    """An error the LLM should see and can recover from."""


def plain_text_to_html(text: str) -> str:
    paragraphs = [escape(line) for line in (text or "").split("\n")]
    return "".join(f"<p>{line}</p>" if line else "<p></p>" for line in paragraphs)


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except (TypeError, ValueError):
        return False


class WorkItemTools:
    """Executes the LLM's tool calls on behalf of ``user`` inside ``workspace``."""

    def __init__(self, user, workspace: Workspace, origin: Optional[str] = None):
        self.user = user
        self.workspace = workspace
        self.origin = origin
        # Work items created or changed during this chat turn, for the client to refresh
        self.actions: List[Dict[str, Any]] = []
        self._handlers: Dict[str, Callable[..., Any]] = {
            "list_projects": self.list_projects,
            "get_project_details": self.get_project_details,
            "search_work_items": self.search_work_items,
            "get_work_item": self.get_work_item,
            "create_work_item": self.create_work_item,
            "update_work_item": self.update_work_item,
            "add_comment": self.add_comment,
        }

    # ---------------------------------------------------------------- dispatch

    def execute(self, name: str, raw_arguments: str) -> str:
        """Run one tool call and return its JSON-encoded result."""
        handler = self._handlers.get(name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            arguments = json.loads(raw_arguments or "{}")
            if not isinstance(arguments, dict):
                raise ToolError("Tool arguments must be a JSON object")
            result = handler(**arguments)
        except (ToolError, json.JSONDecodeError) as e:
            return json.dumps({"error": str(e)})
        except (TypeError, ValueError, DjangoValidationError) as e:
            # Unexpected/missing keyword arguments or malformed values (e.g. dates) from the model
            return json.dumps({"error": f"Invalid arguments: {e}"})
        except Exception as e:
            log_exception(e)
            return json.dumps({"error": "The action failed due to an internal error"})
        return json.dumps(result, cls=DjangoJSONEncoder)

    # ----------------------------------------------------------------- helpers

    def _memberships(self):
        return ProjectMember.objects.filter(
            workspace=self.workspace,
            member=self.user,
            is_active=True,
            role__gte=ROLE.MEMBER.value,
            project__archived_at__isnull=True,
        )

    def _project_ids(self) -> List[uuid.UUID]:
        return list(self._memberships().values_list("project_id", flat=True))

    def _get_project(self, project_id: str) -> Project:
        if not _is_uuid(project_id) or uuid.UUID(str(project_id)) not in self._project_ids():
            raise ToolError("Project not found or you do not have access to it")
        return Project.objects.get(pk=project_id, workspace=self.workspace)

    def _work_items(self):
        return Issue.issue_objects.filter(workspace=self.workspace, project_id__in=self._project_ids())

    def _get_work_item(self, ref: str) -> Issue:
        ref = (ref or "").strip()
        queryset = self._work_items().select_related("project", "state")
        issue = None
        if _is_uuid(ref):
            issue = queryset.filter(pk=ref).first()
        else:
            match = WORK_ITEM_REF_PATTERN.match(ref)
            if match:
                issue = queryset.filter(
                    project__identifier__iexact=match.group(1),
                    sequence_id=int(match.group(2)),
                ).first()
        if issue is None:
            raise ToolError(f"Work item '{ref}' not found or you do not have access to it")
        return issue

    @staticmethod
    def _reference(issue: Issue) -> str:
        return f"{issue.project.identifier}-{issue.sequence_id}"

    def _summaries(self, issues: List[Issue]) -> List[Dict[str, Any]]:
        assignees: Dict[uuid.UUID, List[str]] = {}
        for row in IssueAssignee.objects.filter(issue_id__in=[i.id for i in issues]).values(
            "issue_id", "assignee__display_name"
        ):
            assignees.setdefault(row["issue_id"], []).append(row["assignee__display_name"])
        return [
            {
                "id": issue.id,
                "reference": self._reference(issue),
                "name": issue.name,
                "project": issue.project.name,
                "state": issue.state.name if issue.state else None,
                "state_group": issue.state.group if issue.state else None,
                "priority": issue.priority,
                "assignees": assignees.get(issue.id, []),
                "start_date": issue.start_date,
                "target_date": issue.target_date,
            }
            for issue in issues
        ]

    def _record_action(self, action: str, issue: Issue) -> Dict[str, Any]:
        entry = {
            "action": action,
            "id": str(issue.id),
            "project_id": str(issue.project_id),
            "reference": self._reference(issue),
            "name": issue.name,
        }
        self.actions.append(entry)
        return entry

    def _build_payload(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Translate tool arguments into an IssueCreateSerializer payload."""
        unknown = set(fields) - set(_WORK_ITEM_FIELDS)
        if unknown:
            raise ToolError(f"Unknown fields: {', '.join(sorted(unknown))}")
        payload = {key: value for key, value in fields.items() if key not in ("description", "parent")}
        if "description" in fields:
            payload["description_html"] = plain_text_to_html(fields["description"])
        if "parent" in fields:
            payload["parent_id"] = self._get_work_item(fields["parent"]).id if fields["parent"] else None
        return payload

    def _annotated_work_item(self, issue: Issue) -> Issue:
        """Load the work item with the id lists the activity tracker diffs against."""
        return (
            Issue.issue_objects.filter(pk=issue.pk)
            .annotate(
                label_ids=Coalesce(
                    ArrayAgg(
                        "labels__id",
                        distinct=True,
                        filter=Q(~Q(labels__id__isnull=True) & Q(label_issue__deleted_at__isnull=True)),
                    ),
                    Value([], output_field=ArrayField(UUIDField())),
                ),
                assignee_ids=Coalesce(
                    ArrayAgg(
                        "assignees__id",
                        distinct=True,
                        filter=Q(~Q(assignees__id__isnull=True) & Q(issue_assignee__deleted_at__isnull=True)),
                    ),
                    Value([], output_field=ArrayField(UUIDField())),
                ),
                module_ids=Coalesce(
                    ArrayAgg(
                        "issue_module__module_id",
                        distinct=True,
                        filter=Q(~Q(issue_module__module_id__isnull=True) & Q(issue_module__deleted_at__isnull=True)),
                    ),
                    Value([], output_field=ArrayField(UUIDField())),
                ),
            )
            .first()
        )

    # ------------------------------------------------------------------- tools

    def list_projects(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": membership.project_id,
                "name": membership.project.name,
                "identifier": membership.project.identifier,
                "role": "admin" if membership.role == ROLE.ADMIN.value else "member",
            }
            for membership in self._memberships().select_related("project").order_by("project__name")
        ]

    def get_project_details(self, project_id: str) -> Dict[str, Any]:
        project = self._get_project(project_id)
        return {
            "id": project.id,
            "name": project.name,
            "identifier": project.identifier,
            "states": list(
                State.objects.filter(project=project)
                .exclude(group="triage")
                .order_by("sequence")
                .values("id", "name", "group", "default")
            ),
            "labels": list(Label.objects.filter(project=project).order_by("name").values("id", "name")),
            "members": [
                {
                    "id": member["member_id"],
                    "display_name": member["member__display_name"],
                    "full_name": f"{member['member__first_name']} {member['member__last_name']}".strip(),
                }
                for member in ProjectMember.objects.filter(project=project, is_active=True).values(
                    "member_id",
                    "member__display_name",
                    "member__first_name",
                    "member__last_name",
                )
            ],
        }

    def search_work_items(
        self,
        query: Optional[str] = None,
        project_id: Optional[str] = None,
        assignee_id: Optional[str] = None,
        state_groups: Optional[List[str]] = None,
        priorities: Optional[List[str]] = None,
        due_before: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        queryset = self._work_items().select_related("project", "state")
        if project_id:
            queryset = queryset.filter(project=self._get_project(project_id))
        if query:
            text_filter = Q(name__icontains=query)
            match = WORK_ITEM_REF_PATTERN.match(query.strip())
            if match:
                text_filter |= Q(project__identifier__iexact=match.group(1), sequence_id=int(match.group(2)))
            queryset = queryset.filter(text_filter)
        if assignee_id:
            user_id = self.user.id if assignee_id == "me" else assignee_id
            if not _is_uuid(str(user_id)):
                raise ToolError("assignee_id must be a user UUID or 'me'")
            queryset = queryset.filter(issue_assignee__assignee_id=user_id, issue_assignee__deleted_at__isnull=True)
        if state_groups:
            queryset = queryset.filter(state__group__in=state_groups)
        if priorities:
            queryset = queryset.filter(priority__in=priorities)
        if due_before:
            queryset = queryset.filter(target_date__lte=due_before)

        limit = max(1, min(int(limit or 20), MAX_SEARCH_LIMIT))
        total = queryset.distinct().count()
        issues = list(queryset.distinct().order_by("-updated_at")[:limit])
        return {"total": total, "results": self._summaries(issues)}

    def get_work_item(self, work_item: str) -> Dict[str, Any]:
        issue = self._get_work_item(work_item)
        details = self._summaries([issue])[0]
        details.update(
            {
                "description": (issue.description_stripped or "")[:4000],
                "project_id": issue.project_id,
                "state_id": issue.state_id,
                "assignee_ids": list(IssueAssignee.objects.filter(issue=issue).values_list("assignee_id", flat=True)),
                "labels": list(IssueLabel.objects.filter(issue=issue).values("label_id", "label__name")),
                "parent": self._reference(issue.parent) if issue.parent_id and issue.parent else None,
                "created_at": issue.created_at,
                "updated_at": issue.updated_at,
            }
        )
        return details

    def create_work_item(self, project_id: str, **fields: Any) -> Dict[str, Any]:
        project = self._get_project(project_id)
        payload = self._build_payload(fields)
        serializer = IssueCreateSerializer(
            data=payload,
            context={
                "project_id": project.id,
                "workspace_id": project.workspace_id,
                "default_assignee_id": project.default_assignee_id,
            },
        )
        if not serializer.is_valid():
            raise ToolError(f"Validation failed: {json.dumps(serializer.errors, cls=DjangoJSONEncoder)}")
        serializer.save()
        issue = Issue.objects.select_related("project").get(pk=serializer.data["id"])

        issue_activity.delay(
            type="issue.activity.created",
            requested_data=json.dumps(payload, cls=DjangoJSONEncoder),
            actor_id=str(self.user.id),
            issue_id=str(issue.id),
            project_id=str(project.id),
            current_instance=None,
            epoch=int(timezone.now().timestamp()),
            notification=True,
            origin=self.origin,
        )
        model_activity.delay(
            model_name="issue",
            model_id=str(issue.id),
            requested_data=payload,
            current_instance=None,
            actor_id=self.user.id,
            slug=self.workspace.slug,
            origin=self.origin,
        )
        issue_description_version_task.delay(
            updated_issue=json.dumps(payload, cls=DjangoJSONEncoder),
            issue_id=str(issue.id),
            user_id=self.user.id,
            is_creating=True,
        )
        return self._record_action("created", issue)

    def update_work_item(self, work_item: str, **fields: Any) -> Dict[str, Any]:
        issue = self._get_work_item(work_item)
        payload = self._build_payload(fields)
        if not payload:
            raise ToolError("No fields to update")

        current_instance = json.dumps(
            IssueDetailSerializer(self._annotated_work_item(issue)).data, cls=DjangoJSONEncoder
        )
        serializer = IssueCreateSerializer(issue, data=payload, partial=True, context={"project_id": issue.project_id})
        if not serializer.is_valid():
            raise ToolError(f"Validation failed: {json.dumps(serializer.errors, cls=DjangoJSONEncoder)}")
        serializer.save()

        issue_activity.delay(
            type="issue.activity.updated",
            requested_data=json.dumps(payload, cls=DjangoJSONEncoder),
            actor_id=str(self.user.id),
            issue_id=str(issue.id),
            project_id=str(issue.project_id),
            current_instance=current_instance,
            epoch=int(timezone.now().timestamp()),
            notification=True,
            origin=self.origin,
        )
        model_activity.delay(
            model_name="issue",
            model_id=str(issue.id),
            requested_data=payload,
            current_instance=current_instance,
            actor_id=self.user.id,
            slug=self.workspace.slug,
            origin=self.origin,
        )
        issue_description_version_task.delay(
            updated_issue=current_instance,
            issue_id=str(issue.id),
            user_id=self.user.id,
        )
        issue.refresh_from_db()
        return self._record_action("updated", issue)

    def add_comment(self, work_item: str, comment: str) -> Dict[str, Any]:
        issue = self._get_work_item(work_item)
        if not (comment or "").strip():
            raise ToolError("Comment cannot be empty")
        serializer = IssueCommentSerializer(data={"comment_html": plain_text_to_html(comment)})
        if not serializer.is_valid():
            raise ToolError(f"Validation failed: {json.dumps(serializer.errors, cls=DjangoJSONEncoder)}")
        serializer.save(project_id=issue.project_id, issue_id=issue.id, actor=self.user)

        issue_activity.delay(
            type="comment.activity.created",
            requested_data=json.dumps(serializer.data, cls=DjangoJSONEncoder),
            actor_id=str(self.user.id),
            issue_id=str(issue.id),
            project_id=str(issue.project_id),
            current_instance=None,
            epoch=int(timezone.now().timestamp()),
            notification=True,
            origin=self.origin,
        )
        model_activity.delay(
            model_name="issue_comment",
            model_id=str(serializer.data["id"]),
            requested_data={"comment_html": serializer.data["comment_html"]},
            current_instance=None,
            actor_id=self.user.id,
            slug=self.workspace.slug,
            origin=self.origin,
        )
        return self._record_action("commented", issue)
