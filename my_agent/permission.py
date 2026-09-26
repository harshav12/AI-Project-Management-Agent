import json
from enum import Enum
from pathlib import Path


class Permission(str, Enum):
    VIEW_PROJECT = "VIEW_PROJECT"
    VIEW_TASK = "VIEW_TASK"
    CREATE_TASK = "CREATE_TASK"
    UPDATE_TASK = "UPDATE_TASK"
    VIEW_EMPLOYEE = "VIEW_EMPLOYEE"
    VIEW_PROJECT_METRICS = "VIEW_PROJECT_METRICS"


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