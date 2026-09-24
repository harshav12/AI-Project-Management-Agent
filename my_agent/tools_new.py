import json


import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

with open(DATA_DIR / "employees.json", "r", encoding="utf-8") as file:
    employees = json.load(file)

with open(DATA_DIR / "projects.json", "r", encoding="utf-8") as file:
    projects = json.load(file)

with open(DATA_DIR / "tasks.json", "r", encoding="utf-8") as file:
    tasks = json.load(file)

with open(DATA_DIR / "project_updates.json", "r", encoding="utf-8") as file:
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
def get_employee(employee_id=None, employee_name=None):
    """Return one employee by ID or name."""

    if employee_id is None and employee_name is None:
        raise ValueError(
            "Provide employee_id or employee_name."
        )

    # --------------------------------------------------------
    # Search by employee ID
    # --------------------------------------------------------

    if employee_id is not None:
        normalized_id = str(employee_id).strip().upper()

        for employee in employees:
            if (
                str(employee.get("employee_id", "")).upper()
                == normalized_id
            ):
                return employee

    # --------------------------------------------------------
    # Search by employee name
    # --------------------------------------------------------

    if employee_name is not None:
        normalized_name = (
            str(employee_name).strip().lower()
        )

        for employee in employees:
            employee_name_value = (
                str(employee.get("name", "")).strip().lower()
            )

            if employee_name_value == normalized_name:
                return employee

    return None

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

