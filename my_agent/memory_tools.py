import json
from pathlib import Path
from datetime import date

FILE_PATH = Path(__file__).resolve()

PROJECT_ROOT = FILE_PATH.parent.parent
MEMORY_DIR = PROJECT_ROOT / "Memory"

USERS_FILE = MEMORY_DIR / "users.json"
USER_MEMORY_FILE = MEMORY_DIR / "user_memory.json"
PROJECTS_FILE = PROJECT_ROOT / "my_agent" / "projects.json"

with open(USERS_FILE, "r", encoding = "utf-8") as file:
    users = json.load(file)

with open(USER_MEMORY_FILE, "r", encoding = "utf-8") as file:
    user_memories = json.load(file)

with open(PROJECTS_FILE, "r", encoding = "utf-8") as file:
    projects = json.load(file)


# get users
def get_user(user_id):
    "Return one user by user ID. "
    for user in users:
        if user.get("user_id") == user_id:
            return user

    return None

# get active memories

def get_active_memories(user_id):
    """Return only active memories belonging to one user. """
    user = get_user(user_id)

    if user is None:
        raise ValueError(f"unknown user_id: {user_id}")

    return [
        memory
        for memory in user_memories
        if memory.get("user_id") == user_id 
        and memory.get("status") == "active"
    ]

# for relevant memories
def get_relevant_memories(user_id, project_id = None, task_id = None):
    """Return active memories relevant to a project or task."""
    active_memories = get_active_memories(user_id)

    if project_id is None and task_id is None:
        return active_memories

    relevant_memories = []

    for memory in active_memories:
        if project_id is not None:
            if(
                memory.get("subject_type") == "project"
                and memory.get("subject_id") == project_id ):

                relevant_memories.append(memory)
            
        if task_id is not None:
            if(
                memory.get("subject_type") == "task"
                and memory.get("subject_id") == task_id ):
                relevant_memories.append(memory)
    
    return relevant_memories


# get active memories relevant to a natural language request

def get_relevant_memories_for_query(user_id, user_query):

    """Find active memories relevant to a natural-language request."""
    active_memories = get_active_memories(user_id)
    query = user_query.lower()

    project_id = find_project_id_in_query(user_query)

    matched_subject_ids = set()

    if project_id:
        matched_subject_ids.add(project_id)

    for memory in active_memories:
        subject_id = str(memory.get("subject_id", "")).lower()

        if subject_id and subject_id in query:
            matched_subject_ids.add(subject_id.upper())

    relevant_memories = [
        memory
        for memory in active_memories
        if str(memory.get("subject_id", "")).upper()
        in matched_subject_ids
    ]

    preference_request = any(
        phrase in query
        for phrase in [
            "priority",
            "priorities",
            "focus",
            "focus on",
            "today",
        ]
    )

    if preference_request:
        relevant_memories.extend(
            memory
            for memory in active_memories
            if memory.get("memory_type") == "preference"
            and memory not in relevant_memories
        )

    return relevant_memories



def find_project_id_in_query(user_query):
    """Return a project ID mentioned by ID or project name."""
    query = user_query.lower()

    for project in projects:
        project_id = project.get("project_id", "").lower()
        project_name = project.get("project_name", "").lower()

        if project_id in query:
            return project_id.upper()

        if project_name in query:
            return project_id.upper()

    return None


# get conversation context

def get_conversation_context(user_id, user_query):
    """Return context for references such as 'that project' or 'the previous task'."""
    query = user_query.lower()

    reference_phrases = [
        "that project",
        "the previous project",
        "that task",
        "the previous task",
    ]

    contains_reference = any(
        phrase in query
        for phrase in reference_phrases
    )

    if not contains_reference:
        return []

    active_memories = get_active_memories(user_id)

    context_memories = [
        memory
        for memory in active_memories
        if memory.get("memory_type") == "conversation_context"
    ]

    if not context_memories:
        return []

    if "project" in query:
        context_memories = [
            memory
            for memory in context_memories
            if memory.get("subject_type") == "project"
        ]

    elif "task" in query:
        context_memories = [
            memory
            for memory in context_memories
            if memory.get("subject_type") == "task"
        ]

    context_memories.sort(
        key=lambda memory: memory.get("updated_at", ""),
        reverse=True,
    )

    return context_memories[:1]


# get memory context
def get_memory_context(user_id, user_query):
    """Build the relevant memory context for one user request."""
    user = get_user(user_id)

    if user is None:
        raise ValueError(f"Unknown user_id: {user_id}")

    direct_memories = get_relevant_memories_for_query(
        user_id=user_id,
        user_query=user_query,
    )

    conversation_memories = get_conversation_context(
        user_id=user_id,
        user_query=user_query,
    )

    combined_memories = []

    for memory in direct_memories + conversation_memories:
        if memory not in combined_memories:
            combined_memories.append(memory)

    return {
        "user_id": user_id,
        "memories": combined_memories,
    }


# update preference

def update_preference(
    user_id,
    project_id,
    value,
    status="active",
):
    """Create or update a user's project preference."""

    user = get_user(user_id)

    if user is None:
        raise ValueError(f"Unknown user_id: {user_id}")

    matching_memory = None

    for memory in user_memories:
        if (
            memory.get("user_id") == user_id
            and memory.get("subject_type") == "project"
            and memory.get("subject_id") == project_id
            and memory.get("memory_type") == "preference"
        ):
            matching_memory = memory
            break

    today = date.today().isoformat()

    if matching_memory is not None:
        previous_value = matching_memory.get("value")
        previous_status = matching_memory.get("status")

        matching_memory["value"] = value
        matching_memory["status"] = status
        matching_memory["updated_at"] = today

        action = "updated"

    else:
        numeric_ids = [
            int(memory["memory_id"][1:])
            for memory in user_memories
            if str(memory.get("memory_id", "")).startswith("M")
            and memory["memory_id"][1:].isdigit()
        ]

        next_memory_number = max(numeric_ids, default=0) + 1

        matching_memory = {
            "memory_id": f"M{next_memory_number:03d}",
            "user_id": user_id,
            "memory_type": "preference",
            "subject_type": "project",
            "subject_id": project_id,
            "value": value,
            "status": status,
            "updated_at": today,
        }

        user_memories.append(matching_memory)

        previous_value = None
        previous_status = None
        action = "created"

    with USER_MEMORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(user_memories, file, indent=2)

    return {
        "action": action,
        "memory_id": matching_memory["memory_id"],
        "user_id": user_id,
        "project_id": project_id,
        "previous_value": previous_value,
        "previous_status": previous_status,
        "new_value": value,
        "new_status": status,
        "updated_at": today,
    }


# Process preference update
def process_preference_update(user_id, user_query):
    """Detect and apply an explicit project-priority update."""

    query = user_query.lower()
    project_id = find_project_id_in_query(user_query)

    if project_id is None:
        return None

    if (
        "no longer" in query
        and "priority" in query
    ):
        return update_preference(
            user_id=user_id,
            project_id=project_id,
            value="not a priority",
            status="inactive",
        )

    if (
        "remember that" in query
        and "priority" in query
    ):
        return update_preference(
            user_id=user_id,
            project_id=project_id,
            value="priority",
            status="active",
        )

    return None