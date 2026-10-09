import json


import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

EMPLOYEES_FILE = DATA_DIR / "employees.json"
PROJECTS_FILE = DATA_DIR / "projects.json"
TASKS_FILE = DATA_DIR / "tasks.json"
PROJECT_UPDATES_FILE = DATA_DIR / "project_updates.json"


with open(EMPLOYEES_FILE, "r", encoding="utf-8") as file:
    employees = json.load(file)

with open(PROJECTS_FILE, "r", encoding="utf-8") as file:
    projects = json.load(file)

with open(TASKS_FILE, "r", encoding="utf-8") as file:
    tasks = json.load(file)

with open(PROJECT_UPDATES_FILE, "r", encoding="utf-8") as file:
    project_updates = json.load(file)


# get projects
def get_projects(status=None):
    """Return projects, optionally filtered by status."""

    if status is None:
        return projects

    requested_status = status.strip().lower()

    return [
        project
        for project in projects
        if str(project.get("status", "")).lower() == requested_status
    ]



# get project


def get_project(
    project_id: str | None = None,
    project_name: str | None = None
):
    """Look up a project by exact ID or by a project name.

    Rules:
    - Use project_id for IDs like "P004".
    - Use project_name for names like "Project Delta" or short forms like "Delta".
    - Partial project names are supported and will be resolved automatically.
    - If both are provided, project_id is checked first, then project_name.
    """
    if project_id is None and project_name is None:
        raise ValueError("Provide project_id or project_name.")

    raw_project_id = str(project_id).strip() if project_id is not None else None
    raw_project_name = str(project_name).strip() if project_name is not None else None

    # exact project_id match
    if raw_project_id:
        normalized_id = raw_project_id.upper()
        for project in projects:
            if str(project.get("project_id", "")).upper() == normalized_id:
                return project

    # exact project_name match
    if raw_project_name:
        normalized_name = raw_project_name.lower().strip()
        for project in projects:
            project_name_value = str(project.get("project_name", "")).lower().strip()
            if project_name_value == normalized_name:
                return project

    # partial match: "Delta" or "Project Delta"
    candidates = []
    if raw_project_id:
        candidates.append(raw_project_id)
    if raw_project_name:
        candidates.append(raw_project_name)

    for candidate in candidates:
        normalized_candidate = candidate.strip().lower().replace("project ", "").strip()

        for project in projects:
            project_name_value = str(project.get("project_name", "")).lower().strip()
            project_name_short = project_name_value.replace("project ", "").strip()

            if (
                normalized_candidate == project_name_short
                or normalized_candidate == project_name_value
                or normalized_candidate in project_name_value
                or project_name_value.endswith(normalized_candidate)
                or project_name_short.endswith(normalized_candidate)
            ):
                return project

    return None

# get tasks
# get tasks
def get_tasks(project_id=None, status=None):
    """Return tasks, optionally filtered by project ID and status."""

    filtered_tasks = tasks

    # Filter by project ID if provided
    if project_id is not None:
        normalized_id = project_id.strip().upper()

        if not normalized_id.startswith("P"):
            raise ValueError(
                "project_id must be a valid ID such as P001."
            )

        filtered_tasks = [
            task
            for task in filtered_tasks
            if str(task.get("project_id", "")).upper()
            == normalized_id
        ]

    # Filter by task status if provided
    if status is not None:
        normalized_status = (
            status.strip()
            .lower()
            .replace(" ", "_")
        )

        filtered_tasks = [
            task
            for task in filtered_tasks
            if str(task.get("status", "")).strip().lower()
            == normalized_status
        ]

    return filtered_tasks



# ============================================================
# TASK UPDATE TOOLS
# ============================================================

VALID_TASK_STATUSES = {
    "pending",
    "in_progress",
    "completed",
    "overdue",
}


def _save_tasks():
    """Persist the current task data to tasks.json."""

    with open(TASKS_FILE, "w", encoding="utf-8") as file:
        json.dump(tasks, file, indent=2)


def get_task(task_id):
    """Return one task by task ID."""

    if task_id is None:
        return None

    normalized_id = str(task_id).strip().upper()

    for task in tasks:
        if str(task.get("task_id", "")).upper() == normalized_id:
            return task

    return None


def update_task_status(task_id, status):
    """
    Update the status of an existing task and persist the change.
    """

    task = get_task(task_id)

    if task is None:
        return {
            "success": False,
            "error": f"Task {task_id} was not found."
        }

    normalized_status = (
        str(status)
        .strip()
        .lower()
        .replace(" ", "_")
    )

    if normalized_status not in VALID_TASK_STATUSES:
        return {
            "success": False,
            "error": (
                f"Invalid status '{status}'. "
                f"Valid statuses are: "
                f"{', '.join(sorted(VALID_TASK_STATUSES))}."
            )
        }

    old_status = task.get("status")

    task["status"] = normalized_status

    _save_tasks()

    return {
        "success": True,
        "task_id": task["task_id"],
        "old_status": old_status,
        "new_status": normalized_status,
    }


def assign_task(task_id, employee_id):
    """
    Assign or reassign an existing task to an employee.
    """

    task = get_task(task_id)

    if task is None:
        return {
            "success": False,
            "error": f"Task {task_id} was not found."
        }

    if employee_id is None:
        return {
            "success": False,
            "error": "An employee ID is required."
        }

    normalized_employee_id = str(employee_id).strip().upper()

    employee = get_employee(employee_id=normalized_employee_id)

    if employee is None:
        return {
            "success": False,
            "error": (
                f"Employee {normalized_employee_id} "
                "was not found."
            )
        }

    old_assignee = task.get("assigned_to")

    task["assigned_to"] = normalized_employee_id

    _save_tasks()

    return {
        "success": True,
        "task_id": task["task_id"],
        "old_assignee": old_assignee,
        "new_assignee": normalized_employee_id,
        "employee_name": employee.get("name"),
    }



# get project updates
def get_project_updates(project_id=None):
    """Return all project updates or updates for one project."""

    if project_id is None:
        return project_updates

    normalized_id = project_id.strip().upper()

    if not normalized_id.startswith("P"):
        raise ValueError("project_id must be a valid ID such as P001.")

    return [
        update
        for update in project_updates
        if str(update.get("project_id", "")).upper() == normalized_id
    ]



# get employee
def get_employee(
    employee_id=None,
    employee_name=None,
    department=None
):
    """Retrieve one or more employees using optional filters."""

    filtered_employees = employees

    # --------------------------------------------------------
    # Filter by employee ID
    # --------------------------------------------------------

    if employee_id is not None:
        normalized_id = str(employee_id).strip().upper()

        filtered_employees = [
            employee
            for employee in filtered_employees
            if str(employee.get("employee_id", "")).strip().upper()
            == normalized_id
        ]

    # --------------------------------------------------------
    # Filter by employee name
    # --------------------------------------------------------

    if employee_name is not None:
        normalized_name = str(employee_name).strip().lower()

        filtered_employees = [
            employee
            for employee in filtered_employees
            if normalized_name
            in str(employee.get("name", "")).strip().lower()
        ]

    # --------------------------------------------------------
    # Filter by department
    # --------------------------------------------------------

    if department is not None:
        normalized_department = str(department).strip().lower()

        filtered_employees = [
            employee
            for employee in filtered_employees
            if str(employee.get("department", "")).strip().lower()
            == normalized_department
        ]

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    if employee_id is not None:
        return filtered_employees[0] if filtered_employees else None

    if employee_name is not None and len(filtered_employees) == 1:
        return filtered_employees[0]

    return filtered_employees

# get project metrics

def get_project_metrics(project_id):
    """Calculate task metrics and data-quality indicators for a project."""

    project = get_project(project_id=project_id)

    if project is None:
        return {
            "success": False,
            "project_id": project_id,
            "error": f"Project {project_id} was not found."
        }

    project_tasks = get_tasks(project_id=project_id)

    counts = {
        "total": len(project_tasks),
        "completed": 0,
        "in_progress": 0,
        "pending": 0,
        "overdue": 0,
        "other_status": 0
    }

    missing_assignees = []
    invalid_assignees = []
    missing_fields = []

    valid_employee_ids = {
        str(employee.get("employee_id", "")).upper()
        for employee in employees
    }

    required_fields = [
        "task_id",
        "project_id",
        "assigned_to",
        "title",
        "status",
        "due_date"
    ]

    for task in project_tasks:
        status = str(task.get("status", "")).lower()

        if status in counts:
            counts[status] += 1
        else:
            counts["other_status"] += 1

        assigned_to = task.get("assigned_to")

        if not assigned_to:
            missing_assignees.append(task.get("task_id"))
        elif str(assigned_to).upper() not in valid_employee_ids:
            invalid_assignees.append({
                "task_id": task.get("task_id"),
                "assigned_to": assigned_to
            })

        task_missing_fields = [
            field for field in required_fields
            if field not in task or task[field] in (None, "")
        ]

        if task_missing_fields:
            missing_fields.append({
                "task_id": task.get("task_id"),
                "fields": task_missing_fields
            })

    completed = counts["completed"]
    total = counts["total"]

    completion_rate = round((completed / total) * 100, 2) if total else 0

    manager_id = project.get("manager_id")
    manager_valid = (
        manager_id is not None
        and str(manager_id).upper() in valid_employee_ids
    )

    return {
        "success": True,
        "project_id": project_id.upper(),
        "project_name": project.get("project_name"),
        "task_counts": counts,
        "completion_rate": completion_rate,
        "missing_assignees": missing_assignees,
        "invalid_assignees": invalid_assignees,
        "missing_task_fields": missing_fields,
        "manager_id": manager_id,
        "manager_exists": manager_valid,
        "has_no_tasks": total == 0
    }

