# It should handle questions like
# Who is assigned to this task
# What is employee's information
# Which employees are working on project gamma


# ============================================================
# EMPLOYEE SPECIALIST AGENT
# ============================================================

import json

from ollama import chat

from my_agent.tools_new import get_employee


# ============================================================
# EMPLOYEE TOOLS
# ============================================================

employee_tools = {
    "get_employee": get_employee
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

EMPLOYEE_SYSTEM_PROMPT = """
You are an Employee Specialist Agent.

Your responsibility is to solve employee-related requests only.

You can analyze:
- employee information
- employee IDs
- employee names
- employee departments
- employee roles
- other employee information returned by the employee tool

You have access only to employee-related tools.

IMPORTANT OPERATING RULES:

1. Understand the user's employee-related request before acting.

2. Determine exactly what information is required to answer the user's
   request. Do not retrieve additional employee information unless it is
   necessary to answer the request.

3. Create a specific plan before using any tool.

4. Execute the plan by selecting exactly ONE tool call at a time.

5. After every tool result, observe and analyze the result.

6. After observing a tool result, decide whether to:
   - CONTINUE: the current plan is still valid and another tool is needed.
   - REPLAN: the current plan is no longer appropriate, so create a revised plan.
   - FINISH: enough information has been collected to answer the request.

7. If a tool result already contains the information required to answer
   the user's request, FINISH instead of calling additional tools.

8. Never repeat an identical tool call.

9. Do not invent employees, employee IDs, names, departments, roles,
   or any other information.

10. Use only information returned by the available employee tools.

11. The get_employee tool supports:
    - employee_id for IDs such as "E001"
    - employee_name for names such as "John Smith"

12. When an employee name is provided, use employee_name.

13. When an employee ID is provided, use employee_id.

14. If the user provides an ambiguous employee name and the available
    tool cannot uniquely identify the employee, report the ambiguity
    instead of guessing.

15. If information is missing, conflicting, or unavailable, explicitly
    report the limitation instead of guessing.

16. If a tool fails, analyze the failure and decide whether the plan needs
    to be revised or whether the task cannot be completed.

17. Do not try to answer questions that belong to other specialized agents,
    such as project-specific or task-specific information.

18. Stop when the user's employee-related objective has been completed.

Your final response must answer the user's request directly and should be
based only on the information collected from the employee tools.
"""


# ============================================================
# OBSERVATION PROMPT
# ============================================================

OBSERVATION_PROMPT = """
You are observing the result of the latest employee tool call.

Current plan:
{current_plan}

Latest tool:
{tool_name}

Tool arguments:
{arguments}

Tool result:
{result}

Your job is to decide whether the latest tool result is sufficient
to answer the user's original request.

IMPORTANT:

1. Focus ONLY on the information actually requested by the user.

2. If the tool result directly contains the information requested by
   the user, choose FINISH.

3. Do NOT require additional information just because it might be useful,
   interesting, or related to the employee.

4. Do NOT require tools, reports, databases, or information that are not
   available in the employee tools.

5. Do NOT invent missing tools or assume that another source exists.

6. Examples:

   - If the user asks for an employee's details and get_employee returns
     the employee information, choose FINISH.

   - If the user asks for an employee's department and get_employee returns
     the department, choose FINISH.

   - If the user asks for an employee's role and get_employee returns the
     role, choose FINISH.

   - If the user asks for an employee by name and get_employee returns the
     matching employee, choose FINISH.

7. CONTINUE should be used ONLY when the requested information cannot yet
   be answered from the information collected so far AND another available
   employee tool can provide the missing information.

8. REPLAN should be used ONLY when the current plan genuinely cannot
   continue as originally planned and a different available employee tool
   or approach is required.

9. Never request the same tool call again.

10. If the requested information is unavailable from the available employee
    tools, choose FINISH and clearly report that limitation rather than
    repeatedly calling tools.

11. If the tool result contains an error, analyze the error and do not
    pretend that the requested information was retrieved.

Return ONLY valid JSON in this format:

{{
    "decision": "FINISH",
    "reason": "The tool result contains the information required to answer the user's request.",
    "revised_plan": ""
}}

The decision must be exactly one of:

CONTINUE
- The user's requested information is not yet available.
- Another available employee tool is genuinely required.

REPLAN
- The current plan cannot be followed effectively.
- A different available employee tool or approach is genuinely required.
- Put the revised plan in "revised_plan".

FINISH
- The user's requested information is already available.
- No additional tool is required.

For FINISH, use:

{{
    "decision": "FINISH",
    "reason": "Why the available information is sufficient.",
    "revised_plan": ""
}}
"""


# ============================================================
# PLAN CREATION
# ============================================================

def create_employee_plan(user_query):

    messages = [
        {
            "role": "system",
            "content": EMPLOYEE_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""
USER REQUEST:
{user_query}

Create a specific step-by-step plan for solving this request.

Do not execute any tools yet.
Return only the plan.
"""
        }
    ]

    response = chat(
        model="llama3.2:3b",
        messages=messages,
        options={
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 500
        },
        keep_alive="10m"
    )

    return response.message.content or ""


# ============================================================
# OBSERVE TOOL RESULT
# ============================================================

def observe_employee_result(
    current_plan,
    tool_name,
    arguments,
    result
):

    prompt = OBSERVATION_PROMPT.format(
        current_plan=current_plan,
        tool_name=tool_name,
        arguments=json.dumps(arguments, default=str),
        result=json.dumps(result, default=str)
    )

    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": EMPLOYEE_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 300
        },
        keep_alive="10m"
    )

    content = response.message.content or ""

    try:
        return json.loads(content)

    except json.JSONDecodeError:

        # If the observation response is invalid,
        # do not assume that the task is finished.

        return {
            "decision": "ERROR",
            "reason": "Observation response could not be parsed safely.",
            "revised_plan": ""
        }


# ============================================================
# FINAL ANSWER
# ============================================================

def create_employee_final_answer(user_query, trace):

    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": EMPLOYEE_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
USER REQUEST:
{user_query}

The Employee Agent has completed its investigation.

Tool execution trace:
{json.dumps(trace, indent=2, default=str)}

Using ONLY the information in the tool execution trace,
provide the final answer to the user's request.

Do not invent information.

If the available information is insufficient, clearly state
what could and could not be determined.
"""
            }
        ],
        options={
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 700
        },
        keep_alive="10m"
    )

    return response.message.content or ""


# ============================================================
# MAIN EMPLOYEE AGENT
# ============================================================

def run_employee_agent(user_query, max_iterations=10):

    # --------------------------------------------------------
    # 1. PLAN
    # --------------------------------------------------------

    current_plan = create_employee_plan(user_query)

    executed_calls = set()
    trace = []

    # --------------------------------------------------------
    # 2. EXECUTION LOOP
    # --------------------------------------------------------

    for iteration in range(max_iterations):

        messages = [
            {
                "role": "system",
                "content": EMPLOYEE_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
USER REQUEST:
{user_query}

CURRENT PLAN:
{current_plan}

Execute the current plan.

Choose exactly ONE useful employee tool call.

If no tool is required, provide the final answer.
"""
            }
        ]

        # Give the agent information about previous tool executions
        if trace:

            messages.append({
                "role": "user",
                "content": f"""
PREVIOUS TOOL EXECUTION TRACE:
{json.dumps(trace, indent=2, default=str)}

Continue from the current state.
Do not repeat an identical tool call.
"""
            })

        # ----------------------------------------------------
        # ASK LLM TO EXECUTE
        # ----------------------------------------------------

        response = chat(
            model="llama3.2:3b",
            messages=messages,
            tools=list(employee_tools.values()),
            options={
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 700
            },
            keep_alive="10m"
        )

        tool_calls = response.message.tool_calls or []

        # ----------------------------------------------------
        # NO TOOL CALL
        # ----------------------------------------------------

        if not tool_calls:

            final_answer = response.message.content or ""

            return {
                "success": True,
                "agent": "employee_agent",
                "plan": current_plan,
                "report": final_answer,
                "trace": trace,
                "iterations": iteration + 1
            }

        # ----------------------------------------------------
        # ONE TOOL CALL ONLY
        # ----------------------------------------------------

        tool_call = tool_calls[0]

        tool_name = tool_call.function.name

        raw_arguments = tool_call.function.arguments or {}

        if isinstance(raw_arguments, str):

            try:
                arguments = json.loads(raw_arguments)

            except json.JSONDecodeError:

                arguments = {}

        else:

            arguments = dict(raw_arguments)

        call_signature = (
            tool_name,
            json.dumps(arguments, sort_keys=True)
        )

        # ----------------------------------------------------
        # DUPLICATE TOOL CALL PROTECTION
        # ----------------------------------------------------

        if call_signature in executed_calls:

            return {
                "success": False,
                "agent": "employee_agent",
                "plan": current_plan,
                "report": (
                    "The Employee Agent attempted to repeat "
                    "the same tool call and was stopped safely."
                ),
                "trace": trace,
                "iterations": iteration + 1
            }

        else:

            executed_calls.add(call_signature)

            tool_function = employee_tools.get(tool_name)

            if tool_function is None:

                result = {
                    "success": False,
                    "error": f"Unknown employee tool: {tool_name}"
                }

            else:

                try:

                    result = tool_function(**arguments)

                except Exception as error:

                    result = {
                        "success": False,
                        "error": (
                            f"{type(error).__name__}: {error}"
                        )
                    }

        # ----------------------------------------------------
        # RECORD TOOL EXECUTION
        # ----------------------------------------------------

        trace.append({
            "iteration": iteration + 1,
            "tool": tool_name,
            "arguments": arguments,
            "result": result
        })

        # ----------------------------------------------------
        # 3. OBSERVE
        # ----------------------------------------------------

        observation = observe_employee_result(
            current_plan=current_plan,
            tool_name=tool_name,
            arguments=arguments,
            result=result
        )

        decision = observation.get("decision", "ERROR")
        revised_plan = observation.get("revised_plan", "")

        # ----------------------------------------------------
        # 4. DECISION
        # ----------------------------------------------------

        if decision == "FINISH":

            final_answer = create_employee_final_answer(
                user_query=user_query,
                trace=trace
            )

            return {
                "success": True,
                "agent": "employee_agent",
                "plan": current_plan,
                "report": final_answer,
                "trace": trace,
                "iterations": iteration + 1
            }

        elif decision == "REPLAN":

            if revised_plan:

                current_plan = revised_plan

            else:

                current_plan = (
                    current_plan
                    + "\n\nReconsider the remaining objective "
                      "based on the latest tool result."
                )

        elif decision == "CONTINUE":

            continue

        else:

            return {
                "success": False,
                "agent": "employee_agent",
                "plan": current_plan,
                "report": (
                    "The Employee Agent could not safely determine "
                    "the next step."
                ),
                "trace": trace,
                "iterations": iteration + 1
            }

    # --------------------------------------------------------
    # 5. MAX ITERATIONS
    # --------------------------------------------------------

    return {
        "success": False,
        "agent": "employee_agent",
        "plan": current_plan,
        "report": (
            "The Employee Agent stopped because the maximum "
            "iteration limit was reached."
        ),
        "trace": trace,
        "iterations": max_iterations
    }