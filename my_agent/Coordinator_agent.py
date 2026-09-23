import json
import ollama

from my_agent.project_agent import run_project_agent
from my_agent.task_agent import run_task_agent
from my_agent.employee_agent import run_employee_agent

from my_agent.memory_tools import (
    process_preference_update,
    get_memory_context
)


# ============================================================
# COORDINATOR SYSTEM PROMPT
# ============================================================

COORDINATOR_SYSTEM_PROMPT = """
You are the Coordinator Agent of a multi-agent Project Management system.

Your job is to understand the user's request, use relevant user memory,
decide which specialist agent or agents are required, execute them,
and combine their results into one accurate final answer.

AVAILABLE SPECIALIST AGENTS:

1. Project Agent
   Handles:
   - project information
   - project status
   - project budget
   - project deadlines
   - project managers
   - project updates
   - project metrics

2. Task Agent
   Handles:
   - tasks
   - task status
   - task assignments
   - tasks belonging to projects
   - completed/pending/in-progress/overdue tasks
   - task due dates

3. Employee Agent
   Handles:
   - employee information
   - employee ID
   - employee name
   - department
   - role

IMPORTANT MEMORY RULES:

- User memory belongs to a specific user_id.
- Only use memory supplied for the current user.
- Do not assume memory from another user.
- Use memory only when it is relevant to the current request.
- Conversation context may resolve references such as:
  "that project"
  "the previous project"
  "that task"
  "the previous task"
- User preferences may affect requests such as:
  "What should I focus on?"
  "What is my priority?"
- Current explicit user instructions take precedence over old memory.
- Do not mention memory unless it is useful to the answer.

PLANNING RULES:

- Create a plan dynamically based on the user's current request.
- Do not follow a hard-coded workflow.
- Determine what information is required.
- Determine which specialist agent or agents are required.
- Determine dependencies between specialist agents.
- Determine the appropriate order of execution.
- Use information obtained from one specialist when it is required by another.
- Re-plan when a specialist result makes the current plan inappropriate.
- Stop when the user's objective has been completed.

ROUTING RULES:

- Use Project Agent for project-related information.
- Use Task Agent for task-related information.
- Use Employee Agent for employee-related information.
- Use multiple agents when information from multiple domains is required.
- Do not call an unnecessary agent.
- Do not answer domain-specific factual questions yourself when a specialist
  agent can provide the information.
- If a request is genuinely ambiguous and cannot be resolved using memory,
  ask the user for clarification.
- Never invent information.

MULTI-AGENT REASONING:

If one agent provides information required to call another agent,
use the result from the first agent.

Example:

User:
"Who manages Project Alpha and what is their role?"

Possible process:

1. Project Agent → find Project Alpha manager ID.
2. Employee Agent → use that employee ID.
3. Combine both results.

Another example:

User:
"Who is assigned to the tasks in Project Alpha?"

Possible process:

1. Project Agent → resolve Project Alpha to its project ID if necessary.
2. Task Agent → retrieve tasks for that project.
3. Employee Agent → resolve employee IDs when employee information is required.
4. Combine the results.

EXECUTION RULES:

- Use exactly one specialist-agent call at a time.
- Observe the result before deciding what to do next.
- Do not repeat an identical specialist-agent call.
- If the required information has been obtained, finish.
- If another specialist is genuinely required, continue.
- If a specialist fails, analyze the failure before deciding whether to retry.
- Never repeatedly call the same agent with the same arguments.

The final answer must be based only on:
1. the user's request,
2. relevant memory,
3. specialist-agent results.

Do not invent facts.
"""


# ============================================================
# SPECIALIST AGENT WRAPPERS
# ============================================================

def call_project_agent(user_query):
    return run_project_agent(
        user_query=user_query
    )


def call_task_agent(user_query):
    return run_task_agent(
        user_query=user_query
    )


def call_employee_agent(user_query):
    return run_employee_agent(
        user_query=user_query
    )


coordinator_tools = {
    "call_project_agent": call_project_agent,
    "call_task_agent": call_task_agent,
    "call_employee_agent": call_employee_agent
}


# ============================================================
# CREATE COORDINATOR PLAN
# ============================================================

def create_coordinator_plan(
    user_id,
    user_query,
    memory_context
):

    prompt = f"""
USER ID:
{user_id}

USER REQUEST:
{user_query}

RELEVANT MEMORY:
{json.dumps(memory_context, indent=2, default=str)}

Create a short execution plan.

The plan must determine:

1. Whether the supplied memory is relevant.
2. Whether the request can be answered without a specialist agent.
3. Which specialist agent is required.
4. Whether multiple specialist agents are required.
5. Whether one agent's result will be needed by another agent.
6. Whether clarification is required.
7. What information must be obtained before the objective can be completed.

The plan must be dynamically created for the current request.
Do not follow a hard-coded workflow.

Do not invent information.

Return the plan as plain text.
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": COORDINATOR_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 500
        },
        keep_alive="10m"
    )

    return response.message.content or ""


# ============================================================
# OBSERVE COORDINATOR RESULT
# ============================================================

def observe_coordinator_result(
    user_query,
    current_plan,
    memory_context,
    trace
):

    observation_prompt = f"""
You are the observation component of a Coordinator Agent.

ORIGINAL USER REQUEST:
{user_query}

CURRENT COORDINATOR PLAN:
{current_plan}

MEMORY CONTEXT:
{json.dumps(memory_context, indent=2, default=str)}

SPECIALIST EXECUTION TRACE:
{json.dumps(trace, indent=2, default=str)}

Decide what the Coordinator should do next.

Possible outputs:

- FINISH
- CONTINUE
- CLARIFY
- REPLAN

RULES:

1. If the specialist result already answers the original request,
   return FINISH.

2. If the request is clear but more information is still needed,
   return CONTINUE.

3. If the request is genuinely ambiguous,
   return CLARIFY.

4. If the current plan is clearly no longer appropriate,
   return REPLAN.

5. Do not request another specialist if the available result
   already satisfies the user's request.

6. Do not repeat an identical specialist call.

7. Return ONLY valid JSON.

FINISH:
{{
    "decision": "FINISH",
    "reason": "The specialist result contains the requested information.",
    "revised_plan": ""
}}

CONTINUE:
{{
    "decision": "CONTINUE",
    "reason": "The original request is clear, but another specialist is required.",
    "revised_plan": ""
}}

CLARIFY:
{{
    "decision": "CLARIFY",
    "reason": "The request is ambiguous.",
    "revised_plan": ""
}}

REPLAN:
{{
    "decision": "REPLAN",
    "reason": "The current plan is no longer appropriate.",
    "revised_plan": "Provide a revised plan."
}}

Return ONLY valid JSON.
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict coordinator observation module. "
                    "Return only valid JSON."
                )
            },
            {
                "role": "user",
                "content": observation_prompt
            }
        ],
        options={
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 300
        }
    )

    content = (response.message.content or "").strip()

    try:
        observation = json.loads(content)

        valid_decisions = {
            "FINISH",
            "CONTINUE",
            "CLARIFY",
            "REPLAN"
        }

        if observation.get("decision") not in valid_decisions:
            return {
                "decision": "FINISH",
                "reason": "Invalid observation decision. Finish safely using the available result.",
                "revised_plan": ""
            }

        return observation

    except json.JSONDecodeError:

        last_result = trace[-1].get("result", {}) if trace else {}

        if isinstance(last_result, dict):
            if last_result.get("success") is True:
                return {
                    "decision": "FINISH",
                    "reason": "The previous specialist result was successful.",
                    "revised_plan": ""
                }

        return {
            "decision": "CONTINUE",
            "reason": "The result may be incomplete and another specialist may be required.",
            "revised_plan": ""
        }


# ============================================================
# FINAL ANSWER
# ============================================================

def create_coordinator_final_answer(
    user_query,
    memory_context,
    trace
):

    prompt = f"""
USER REQUEST:
{user_query}

RELEVANT MEMORY:
{json.dumps(memory_context, indent=2, default=str)}

SPECIALIST AGENT RESULTS:
{json.dumps(trace, indent=2, default=str)}

The specialist agents have completed their work.

Provide the final answer to the user.

Rules:

- Use only information contained in the user request,
  relevant memory, and specialist-agent results.
- Combine information from multiple agents when necessary.
- Do not mention internal agent names unless useful.
- Do not expose internal planning, execution traces, tool calls,
  observations, or reasoning.
- Do not invent missing information.
- If the available information is insufficient, clearly say so.
- If memory was used to resolve a reference, answer naturally.
- Keep the answer clear and directly relevant to the user's request.
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": COORDINATOR_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
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
# MAIN COORDINATOR
# ============================================================

def run_coordinator(user_id, user_query, max_iterations=10):

    if not user_id or not str(user_id).strip():
        return {
            "success": False,
            "error": "user_id is required."
        }

    if not user_query or not str(user_query).strip():
        return {
            "success": False,
            "error": "user_query is required."
        }

    user_id = str(user_id).strip()
    user_query = str(user_query).strip()

    # ---------------------------------------------------------
    # Memory
    # ---------------------------------------------------------

    preference_update = process_preference_update(
        user_id,
        user_query
    )

    memory_context = get_memory_context(
        user_id,
        user_query
    )

    # ---------------------------------------------------------
    # Coordinator Planning
    # ---------------------------------------------------------

    plan = create_coordinator_plan(
        user_id=user_id,
        user_query=user_query,
        memory_context=memory_context
    )

    trace = []
    called_specialists = set()
    current_plan = plan

    # ---------------------------------------------------------
    # Execution Loop
    # ---------------------------------------------------------

    for iteration in range(1, max_iterations + 1):

        forced_agent = None
        forced_query = user_query

        # -----------------------------------------------------
        # Detect required handoff from previous observation
        # -----------------------------------------------------

        if trace:

            last_observation = trace[-1].get(
                "coordinator_observation",
                {}
            )

            if last_observation.get("decision") == "CONTINUE":

                reason = str(
                    last_observation.get("reason", "")
                ).lower()

                revised_plan = str(
                    last_observation.get("revised_plan", "")
                ).lower()

                combined_handoff_text = (
                    reason + " " + revised_plan
                )

                # Project → Employee handoff
                if "employee agent" in combined_handoff_text:

                    employee_id = None

                    for previous_entry in trace:

                        specialist_result = previous_entry.get(
                            "result",
                            {}
                        )

                        if not isinstance(
                            specialist_result,
                            dict
                        ):
                            continue

                        if specialist_result.get("manager_id"):
                            employee_id = specialist_result.get(
                                "manager_id"
                            )
                            break

                        if specialist_result.get("employee_id"):
                            employee_id = specialist_result.get(
                                "employee_id"
                            )
                            break

                        nested_trace = specialist_result.get(
                            "trace",
                            []
                        )

                        for trace_item in nested_trace:

                            tool_result = trace_item.get(
                                "result",
                                {}
                            )

                            if not isinstance(
                                tool_result,
                                dict
                            ):
                                continue

                            if tool_result.get("manager_id"):
                                employee_id = tool_result.get(
                                    "manager_id"
                                )
                                break

                            if tool_result.get("employee_id"):
                                employee_id = tool_result.get(
                                    "employee_id"
                                )
                                break

                        if employee_id:
                            break

                    if employee_id:

                        forced_agent = "call_employee_agent"

                        forced_query = (
                            f"Retrieve the employee details for "
                            f"employee ID {employee_id}, including "
                            f"the employee's name, department, and role."
                        )

        # -----------------------------------------------------
        # Forced Handoff
        # -----------------------------------------------------

        if forced_agent == "call_employee_agent":

            agent_name = "employee_agent"

            if agent_name in called_specialists:
                break

            called_specialists.add(agent_name)

            specialist_result = call_employee_agent(
                user_query=forced_query
            )

            trace.append({
                "agent": "call_employee_agent",
                "arguments": {
                    "user_query": forced_query
                },
                "result": specialist_result
            })

        else:

            # -------------------------------------------------
            # Dynamic LLM Agent Selection
            # -------------------------------------------------

            response = ollama.chat(
                model="llama3.2:3b",
                messages=[
                    {
                        "role": "system",
                        "content": COORDINATOR_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""
Original User Request:
{user_query}

Current Coordinator Plan:
{current_plan}

Memory Context:
{json.dumps(memory_context, indent=2, default=str)}

Execution Trace:
{json.dumps(trace, indent=2, default=str)}

Choose the next specialist agent only if one is genuinely required.

Available specialist agents:

1. call_project_agent
2. call_task_agent
3. call_employee_agent

Rules:

- Use exactly one specialist tool call at a time.
- Do not repeat an identical specialist call.
- If the previous result already answers the request,
  do not call another specialist.
- If another specialist is genuinely required, call it.
- Use information obtained from previous specialists when
  determining the next required action.
"""
                    }
                ],
                tools=list(coordinator_tools.values()),
                options={
                    "temperature": 0,
                    "num_ctx": 4096,
                    "num_predict": 300
                },
                keep_alive="10m"
            )

            # -------------------------------------------------
            # No Specialist Needed
            # -------------------------------------------------

            if not response.message.tool_calls:

                if trace:
                    final_answer = create_coordinator_final_answer(
                        user_query=user_query,
                        memory_context=memory_context,
                        trace=trace
                    )

                    return {
                        "success": True,
                        "agent": "coordinator",
                        "user_id": user_id,
                        "plan": plan,
                        "memory_context": memory_context,
                        "preference_update": preference_update,
                        "report": final_answer,
                        "trace": trace,
                        "iterations": iteration
                    }

                break

            # -------------------------------------------------
            # Execute Specialist
            # -------------------------------------------------

            tool_call = response.message.tool_calls[0]

            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            call_key = (
                f"{tool_name}:"
                f"{json.dumps(arguments, sort_keys=True)}"
            )

            if call_key in called_specialists:
                break

            called_specialists.add(call_key)

            if tool_name == "call_project_agent":

                specialist_result = call_project_agent(
                    **arguments
                )

            elif tool_name == "call_task_agent":

                specialist_result = call_task_agent(
                    **arguments
                )

            elif tool_name == "call_employee_agent":

                specialist_result = call_employee_agent(
                    **arguments
                )

            else:

                specialist_result = {
                    "success": False,
                    "error": (
                        f"Unknown specialist agent: {tool_name}"
                    )
                }

            trace.append({
                "agent": tool_name,
                "arguments": arguments,
                "result": specialist_result
            })

        # -----------------------------------------------------
        # Observe Specialist Result
        # -----------------------------------------------------

        observation = observe_coordinator_result(
            user_query=user_query,
            current_plan=current_plan,
            memory_context=memory_context,
            trace=trace
        )

        if trace:
            trace[-1]["coordinator_observation"] = observation

        decision = observation.get("decision")

        # -----------------------------------------------------
        # Finish
        # -----------------------------------------------------

        if decision == "FINISH":

            final_answer = create_coordinator_final_answer(
                user_query=user_query,
                memory_context=memory_context,
                trace=trace
            )

            return {
                "success": True,
                "agent": "coordinator",
                "user_id": user_id,
                "plan": plan,
                "memory_context": memory_context,
                "preference_update": preference_update,
                "report": final_answer,
                "trace": trace,
                "iterations": iteration
            }

        # -----------------------------------------------------
        # Clarify
        # -----------------------------------------------------

        if decision == "CLARIFY":

            clarification = observation.get(
                "reason",
                "Could you please clarify your request?"
            )

            return {
                "success": True,
                "agent": "coordinator",
                "user_id": user_id,
                "plan": plan,
                "memory_context": memory_context,
                "preference_update": preference_update,
                "report": clarification,
                "trace": trace,
                "iterations": iteration
            }

        # -----------------------------------------------------
        # Replan
        # -----------------------------------------------------

        if decision == "REPLAN":

            revised_plan = observation.get(
                "revised_plan",
                ""
            )

            current_plan = revised_plan or plan
            continue

        # -----------------------------------------------------
        # Continue
        # -----------------------------------------------------

        if decision == "CONTINUE":

            revised_plan = observation.get(
                "revised_plan",
                ""
            )

            if revised_plan:
                current_plan = revised_plan

            continue

        # Unknown decision
        break

    # ---------------------------------------------------------
    # Final Fallback
    # ---------------------------------------------------------

    final_answer = create_coordinator_final_answer(
        user_query=user_query,
        memory_context=memory_context,
        trace=trace
    )

    return {
        "success": True,
        "agent": "coordinator",
        "user_id": user_id,
        "plan": plan,
        "memory_context": memory_context,
        "preference_update": preference_update,
        "report": final_answer,
        "trace": trace,
        "iterations": max_iterations
    }