# It should handle questions like
    # Project status
    # Currently active projects
    # highest risk
    # who manages Project Alpha? 



import json

from ollama import chat

from my_agent.tools_new import (
    get_projects,
    get_project,
    get_project_updates,
    get_project_metrics
)





# ============================================================
# PROJECT TOOLS
# ============================================================

project_tools = {
    "get_projects": get_projects,
    "get_project": get_project,
    "get_project_updates": get_project_updates,
    "get_project_metrics": get_project_metrics
}





# ============================================================
# SYSTEM PROMPT
# ============================================================
PROJECT_SYSTEM_PROMPT = """
You are a Project Specialist Agent.

Your responsibility is to solve project-related requests only.

You can analyze:
- project information
- project status
- project managers
- project deadlines
- project budgets
- project updates
- project risks
- project metrics

You have access only to project-related tools.

IMPORTANT OPERATING RULES:

1. Understand the user's project-related request before acting.

2. Determine exactly what information is required to answer the user's
   request. Do not retrieve additional project information unless it is
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

9. Do not invent projects, managers, dates, budgets, risks, metrics,
   updates, or any other information.

10. Use only information returned by the available tools.

11. If the user specifies a particular project, stay within that project.

12. When the user asks about current project risk, use the latest relevant
    project update.

13. If information is missing, conflicting, or unavailable, explicitly report
    the limitation instead of guessing.

14. If a tool fails, analyze the failure and decide whether the plan needs
    to be revised or whether the task cannot be completed.

15. Do not try to answer questions that belong to other specialized agents,
    such as employee-specific or task-specific information.

16. Stop when the user's project-related objective has been completed.

Your final response must answer the user's request directly and should be
based only on the information collected from the project tools.
"""





# ============================================================
# OBSERVATION PROMPT
# ============================================================

OBSERVATION_PROMPT = """
You are observing the result of the latest project tool call.

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
   interesting, or related to the project.

4. Do NOT require tools, dashboards, reports, databases, or information
   that are not available in the project tools.

5. Do NOT invent missing tools or assume that another source exists.

6. Examples:

   - If the user asks for a project's status and get_project returns
     "status", choose FINISH.

   - If the user asks for a project's budget and get_project returns
     "budget", choose FINISH.

   - If the user asks for a project's deadline and get_project returns
     "deadline", choose FINISH.

   - If the user asks for the latest project updates and
     get_project_updates returns the updates, choose FINISH.

   - If the user asks for currently active projects and
     get_projects(status="active") returns the active projects,
     choose FINISH.

   - If the user asks about current project risk and the latest relevant
     project update contains a risk level, choose FINISH.

7. CONTINUE should be used ONLY when the requested information cannot
   yet be answered from the information collected so far AND another
   available project tool can provide the missing information.

8. REPLAN should be used ONLY when the current plan genuinely cannot
   continue as originally planned and a different available project
   tool or approach is required.

9. Never request the same tool call again.

10. If the requested information is unavailable from the available
    project tools, choose FINISH and clearly report that limitation
    rather than repeatedly calling tools.

Return ONLY valid JSON in this format:

{{
    "decision": "FINISH",
    "reason": "The tool result contains the information required to answer the user's request.",
    "revised_plan": ""
}}

The decision must be exactly one of:

CONTINUE
- The user's requested information is not yet available.
- Another available project tool is genuinely required.

REPLAN
- The current plan cannot be followed effectively.
- A different available project tool or approach is genuinely required.
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

def create_project_plan(user_query):

    messages = [
        {
            "role": "system",
            "content": PROJECT_SYSTEM_PROMPT
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
def observe_project_result(
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
                "content": PROJECT_SYSTEM_PROMPT
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

def create_project_final_answer(user_query, trace):

    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": PROJECT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
USER REQUEST:
{user_query}

The Project Agent has completed its investigation.

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
# MAIN PROJECT AGENT
# ============================================================

def run_project_agent(user_query, max_iterations=10):

    # --------------------------------------------------------
    # 1. PLAN
    # --------------------------------------------------------

    current_plan = create_project_plan(user_query)

    print("\n========== PROJECT AGENT PLAN ==========\n")
    print(current_plan)

    executed_calls = set()
    trace = []

    # --------------------------------------------------------
    # 2. EXECUTION LOOP
    # --------------------------------------------------------

    for iteration in range(max_iterations):

        messages = [
            {
                "role": "system",
                "content": PROJECT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
USER REQUEST:
{user_query}

CURRENT PLAN:
{current_plan}

Execute the current plan.

Choose exactly ONE useful project tool call.

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
            tools=list(project_tools.values()),
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
                "agent": "project_agent",
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

            print("\nWARNING: DUPLICATE TOOL CALL")
            print(
                "Project Agent attempted to repeat "
                "the same tool call."
            )
            print("Stopping safely.")

            return {
                "success": False,
                "agent": "project_agent",
                "plan": current_plan,
                "report": (
                    "The Project Agent attempted to repeat "
                    "the same tool call and was stopped safely."
                ),
                "trace": trace,
                "iterations": iteration + 1
            }

        else:

            executed_calls.add(call_signature)

            tool_function = project_tools.get(tool_name)

            if tool_function is None:

                result = {
                    "success": False,
                    "error": f"Unknown project tool: {tool_name}"
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

        print(
            f"\n========== PROJECT AGENT ITERATION "
            f"{iteration + 1} =========="
        )

        print(f"Tool: {tool_name}")
        print(f"Arguments: {arguments}")
        print("Result:")
        print(result)

        # ----------------------------------------------------
        # 3. OBSERVE
        # ----------------------------------------------------

        observation = observe_project_result(
            current_plan=current_plan,
            tool_name=tool_name,
            arguments=arguments,
            result=result
        )

        decision = observation.get("decision", "ERROR")
        reason = observation.get("reason", "")
        revised_plan = observation.get("revised_plan", "")

        print("\n========== PROJECT AGENT OBSERVATION ==========")
        print(f"Decision: {decision}")
        print(f"Reason: {reason}")

        # ----------------------------------------------------
        # 4. DECISION
        # ----------------------------------------------------

        if decision == "FINISH":

            final_answer = create_project_final_answer(
                user_query=user_query,
                trace=trace
            )

            return {
                "success": True,
                "agent": "project_agent",
                "plan": current_plan,
                "report": final_answer,
                "trace": trace,
                "iterations": iteration + 1
            }

        elif decision == "REPLAN":

            if revised_plan:

                current_plan = revised_plan

                print(
                    "\n========== PROJECT AGENT REPLANNED ==========\n"
                )

                print(current_plan)

            else:

                print(
                    "\nWARNING: REPLAN requested but no revised "
                    "plan was provided."
                )

                current_plan = (
                    current_plan
                    + "\n\nReconsider the remaining objective "
                      "based on the latest tool result."
                )

        elif decision == "CONTINUE":

            print(
                "\nPROJECT AGENT: Continuing current plan."
            )

        else:

            print(
                "\nWARNING: Invalid observation decision. "
                "Stopping safely."
            )

            return {
                "success": False,
                "agent": "project_agent",
                "plan": current_plan,
                "report": (
                    "The Project Agent could not safely determine "
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
        "agent": "project_agent",
        "plan": current_plan,
        "report": (
            "The Project Agent stopped because the maximum "
            "iteration limit was reached."
        ),
        "trace": trace,
        "iterations": max_iterations
    }