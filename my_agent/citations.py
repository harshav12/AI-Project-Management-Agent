import re


def build_citation_sources(trace):
    """Build verified citation references from coordinator tool results."""
    citations = []
    seen = set()

    def add(
        source_type,
        source,
        record_id,
        label,
        passage=None,
        page=None,
        page_type=None,
    ):
        key = (source_type, source, record_id)

        if key in seen:
            return

        seen.add(key)
        citation = {
            "id": f"S{len(citations) + 1}",
            "type": source_type,
            "source": source,
            "record_id": record_id,
            "label": label,
        }

        if passage:
            citation["passage"] = passage

        if page is not None:
            citation["page"] = page

        if page_type is not None:
            citation["page_type"] = page_type

        citations.append(citation)

    def add_document_results(results):
        if not isinstance(results, list):
            return

        for item in results:
            if not isinstance(item, dict) or not item.get("text"):
                continue

            filename = item.get("source", "Unknown document")
            scope = item.get("document_scope", "unknown")
            chunk_index = item.get("chunk_index")
            page = item.get("page")
            page_type = item.get("page_type")

            if scope == "knowledge_base":
                source = f"Knowledge_base/{filename}"
                label = f"Knowledge-base passage from {filename}"
            else:
                source = f"Uploaded document: {filename}"
                label = f"Uploaded-document passage from {filename}"

            record_id = f"chunk {chunk_index}"
            if page is not None:
                page_label = "virtual page" if page_type == "virtual" else "page"
                label += f", {page_label} {page}"

            add(
                source_type="document",
                source=source,
                record_id=record_id,
                label=label,
                passage=item["text"],
                page=page,
                page_type=page_type,
            )

    def add_json_results(tool_name, arguments, result):
        arguments = arguments if isinstance(arguments, dict) else {}

        if tool_name == "get_project_metrics":
            if not isinstance(result, dict) or result.get("success") is False:
                return

            project_id = result.get("project_id") or arguments.get("project_id")
            if not project_id:
                return

            add(
                "json",
                "my_agent/projects.json",
                project_id,
                f"Project record {project_id}",
            )
            add(
                "json",
                "my_agent/tasks.json",
                f"project {project_id}",
                f"Task records summarized for project {project_id}",
            )

            manager_id = result.get("manager_id")
            if manager_id:
                add(
                    "json",
                    "my_agent/employees.json",
                    manager_id,
                    f"Manager employee record {manager_id}",
                )
            return

        rows = result if isinstance(result, list) else [result]

        for row in rows:
            if not isinstance(row, dict) or row.get("success") is False:
                continue

            if tool_name in {"get_projects", "get_project"}:
                record_id = row.get("project_id") or arguments.get("project_id")
                if record_id:
                    name = row.get("project_name", "")
                    add(
                        "json",
                        "my_agent/projects.json",
                        record_id,
                        f"Project record {record_id} {name}".strip(),
                    )

            elif tool_name == "get_project_updates":
                record_id = row.get("update_id")
                if record_id:
                    add(
                        "json",
                        "my_agent/project_updates.json",
                        record_id,
                        f"Project update {record_id} for {row.get('project_id', '')}",
                    )

            elif tool_name in {
                "get_tasks",
                "get_task",
                "update_task_status",
                "assign_task",
            }:
                record_id = row.get("task_id") or arguments.get("task_id")
                if record_id:
                    add(
                        "json",
                        "my_agent/tasks.json",
                        record_id,
                        f"Task record {record_id} {row.get('title', '')}".strip(),
                    )

                employee_id = row.get("new_assignee")
                if employee_id:
                    add(
                        "json",
                        "my_agent/employees.json",
                        employee_id,
                        f"Employee record {employee_id}",
                    )

            elif tool_name == "get_employee":
                record_id = row.get("employee_id")
                if record_id:
                    add(
                        "json",
                        "my_agent/employees.json",
                        record_id,
                        f"Employee record {record_id} {row.get('name', '')}".strip(),
                    )

    for coordinator_step in trace:
        if not isinstance(coordinator_step, dict):
            continue

        agent_name = coordinator_step.get("agent")
        result = coordinator_step.get("result")

        if agent_name == "search_outside_knowledge":
            add_document_results(result)
            continue

        if not isinstance(result, dict):
            continue

        for tool_step in result.get("trace", []):
            if not isinstance(tool_step, dict):
                continue

            tool_name = tool_step.get("tool", "")
            arguments = tool_step.get("arguments", {})
            tool_result = tool_step.get("result")

            if tool_name == "search_knowledge_base_tool":
                add_document_results(tool_result)
            else:
                add_json_results(tool_name, arguments, tool_result)

    return citations


def format_answer_citations(answer, citations):
    """Replace internal citation IDs with readable source references."""
    citations_by_id = {
        citation["id"]: citation
        for citation in citations or []
        if isinstance(citation, dict) and citation.get("id")
    }
    displayed_ids = set()

    def replace_marker(match):
        citation_id = match.group(1)
        citation = citations_by_id.get(citation_id)

        if citation is None:
            return match.group(0)

        if citation_id in displayed_ids:
            return ""

        displayed_ids.add(citation_id)

        source = str(citation.get("source", "Unknown source")).strip()
        if source.lower().startswith("uploaded document:"):
            source = source.partition(":")[2].strip()
        else:
            source = source.replace("\\", "/").rsplit("/", 1)[-1]

        record_id = citation.get("record_id")
        page = citation.get("page")
        if page is not None:
            page_label = (
                "virtual page"
                if citation.get("page_type") == "virtual"
                else "page"
            )
            if record_id:
                return (
                    f"[Source: {source}, {page_label} {page}, "
                    f"{record_id}]"
                )
            return f"[Source: {source}, {page_label} {page}]"

        if record_id:
            return f"[Source: {source}, {record_id}]"

        return f"[Source: {source}]"

    formatted_answer = re.sub(r"\[(S\d+)\]", replace_marker, answer or "")
    return re.sub(r"[ \t]+([,.;:])", r"\1", formatted_answer)