import json
from enum import Enum
from pathlib import Path


class Permission(str, Enum):
    VIEW_PROJECT = "VIEW_PROJECT"
    VIEW_TASK = "VIEW_TASK"
    UPDATE_TASK = "UPDATE_TASK"
    VIEW_EMPLOYEE = "VIEW_EMPLOYEE"
    VIEW_PROJECT_METRICS = "VIEW_PROJECT_METRICS"
    VIEW_KNOWLEDGE_BASE = "VIEW_KNOWLEDGE_BASE"


BASE_DIR = Path(__file__).resolve().parent

EMPLOYEES_FILE = BASE_DIR / "employees.json"
ROLES_FILE = BASE_DIR / "roles.json"


def load_employees():
    with open(EMPLOYEES_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def load_roles():
    with open(ROLES_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def get_employee_role(user_id):
    employees = load_employees()

    for employee in employees:
        if employee["employee_id"] == user_id:
            return employee["role"]

    return None


def has_permission(user_id, permission):
    role = get_employee_role(user_id)

    if role is None:
        return False

    roles = load_roles()

    role_data = roles.get(role)

    if role_data is None:
        return False

    permissions = role_data.get("permissions", [])

    return permission.value in permissions


def require_permission(user_id, permission):
    if not has_permission(user_id, permission):
        raise PermissionError(
            f"User {user_id} is not authorized to perform "
            f"'{permission.value}'."
        )

    return True


TOOL_PERMISSIONS = {
    "get_projects": Permission.VIEW_PROJECT,
    "get_project": Permission.VIEW_PROJECT,
    "get_tasks": Permission.VIEW_TASK,
    "get_project_updates": Permission.VIEW_PROJECT,
    "get_employee": Permission.VIEW_EMPLOYEE,
    "get_project_metrics": Permission.VIEW_PROJECT_METRICS,
    "search_knowledge_base_tool": Permission.VIEW_KNOWLEDGE_BASE,
    "update_task_status": Permission.UPDATE_TASK,
    "assign_task": Permission.UPDATE_TASK,
}


def authorize_tool(user_id, tool_name):
    """
    Check whether the user is authorized to use a specific tool.
    """

    permission = TOOL_PERMISSIONS.get(tool_name)

    if permission is None:
        raise PermissionError(
            f"Tool '{tool_name}' is not authorized because "
            f"no permission is defined for it."
        )

    require_permission(user_id, permission)

    return True