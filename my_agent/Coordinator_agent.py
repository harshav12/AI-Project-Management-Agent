# This coordinator receives the original query. 
# This will decide, which agent to use. 




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

ROUTING RULES:

- Use Project Agent for project-only questions.
- Use Task Agent for task-only questions.
- Use Employee Agent for employee-only questions.
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

- Use exactly one specialist-agent tool call at a time.
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
    """
    Call the Project Agent with the user's request.
    """

    return run_project_agent(
        user_query=user_query
    )


def call_task_agent(user_query):
    """
    Call the Task Agent with the user's request.
    """

    return run_task_agent(
        user_query=user_query
    )


def call_employee_agent(user_query):
    """
    Call the Employee Agent with the user's request.
    """

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

# ============================================================
# OBSERVE COORDINATOR RESULT
# ============================================================

def observe_coordinator_result(
    user_query,
    current_plan,
    memory_context,
    trace
):
    """
    Observe specialist-agent results and decide whether the
    Coordinator should continue, replan, finish, or clarify.
    """

    observation_prompt = f"""
You are the observation component of a Coordinator Agent.

ORIGINAL USER REQUEST:
{user_query}

SPECIALIST RESULT:
{json.dumps(trace[-1] if trace else {}, indent=2, default=str)}

Your ONLY job is to determine whether the specialist result
already answers the ORIGINAL USER REQUEST.

IMPORTANT:

- Judge ONLY what the user explicitly asked for.
- Do NOT add extra information that the user did not request.
- Do NOT use extra fields returned by the specialist to create
  new requirements.
- If the requested information is present in the specialist result,
  return FINISH.
- Return CONTINUE only if information explicitly requested by the
  user is still missing.
- Return CLARIFY only if the ORIGINAL USER REQUEST is genuinely
  ambiguous.
- Return REPLAN only if the current approach cannot answer the
  user's request.

Example:

User request:
"What is the budget of Project Alpha?"

Specialist result:
Project Alpha
budget: 180000
manager_id: E005
status: active

Correct decision:
FINISH

Reason:
"The requested budget is present in the specialist result."

Another example:

User request:
"Who manages Project Alpha and what is their role?"

Specialist result:
Project Alpha
manager_id: E005

Correct decision:
CONTINUE

Reason:
"The manager's name and role are explicitly requested but are
not available yet."

Return ONLY valid JSON.

FINISH:
{{
    "decision": "FINISH",
    "reason": "The specialist result contains the information explicitly requested by the user.",
    "revised_plan": ""
}}

CONTINUE:
{{
    "decision": "CONTINUE",
    "reason": "Information explicitly requested by the user is still missing.",
    "revised_plan": ""
}}

CLARIFY:
{{
    "decision": "CLARIFY",
    "reason": "The original user request is genuinely ambiguous.",
    "revised_plan": ""
}}

REPLAN:
{{
    "decision": "REPLAN",
    "reason": "The current execution approach cannot answer the user's request.",
    "revised_plan": ""
}}
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict execution observer. "
                    "The user's original request is the ONLY source "
                    "of required information. "
                    "Never create additional requirements from extra "
                    "fields returned by a specialist. "
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
            "num_predict": 250
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
                "decision": "REPLAN",
                "reason": "Coordinator returned an invalid decision.",
                "revised_plan": ""
            }

        return observation

    except json.JSONDecodeError:

        return {
            "decision": "REPLAN",
            "reason": "Coordinator observation could not be parsed safely.",
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
- Do not expose internal execution details.
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
    """
    Run the Coordinator Agent.

    The Coordinator:
    1. Retrieves user-specific memory.
    2. Creates an execution plan.
    3. Calls specialist agents.
    4. Observes specialist results.
    5. Uses explicit handoffs between specialists when required.
    6. Combines the results into a final answer.
    """

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
    # MEMORY
    # ---------------------------------------------------------

    preference_update = process_preference_update(
        user_id,
        user_query
    )

    memory_context = get_memory_context(
        user_id,
        user_query
    )

    print("\n========== COORDINATOR MEMORY ==========\n")
    print(json.dumps(memory_context, indent=2))

    # ---------------------------------------------------------
    # COORDINATOR PLAN
    # ---------------------------------------------------------

    plan = create_coordinator_plan(
        user_id=user_id,
        user_query=user_query,
        memory_context=memory_context
    )

    print("\n========== COORDINATOR PLAN ==========\n")
    print(plan)

    trace = []
    called_specialists = set()

    current_plan = plan

    # ---------------------------------------------------------
    # EXECUTION LOOP
    # ---------------------------------------------------------

    for iteration in range(1, max_iterations + 1):

        # -----------------------------------------------------
        # Determine which specialist should be called
        # -----------------------------------------------------

        forced_agent = None
        forced_query = user_query

        # If the previous observation explicitly requested a
        # specialist handoff, honor that handoff instead of asking
        # the model to choose the previous specialist again.
        if trace:
            last_observation = trace[-1].get("coordinator_observation", {})

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

                # -------------------------------------------------
                # Project -> Employee handoff
                # -------------------------------------------------

                if "employee agent" in combined_handoff_text:

                    employee_id = None

                    # Look through previous specialist results
                    # for an employee/manager ID.
                    for previous_entry in trace:
                        specialist_result = previous_entry.get(
                            "result",
                            {}
                        )

                        specialist_trace = specialist_result.get(
                            "trace",
                            []
                        )

                        for trace_item in specialist_trace:
                            tool_result = trace_item.get(
                                "result",
                                {}
                            )

                            if isinstance(tool_result, dict):

                                possible_id = (
                                    tool_result.get("manager_id")
                                    or tool_result.get("employee_id")
                                )

                                if possible_id:
                                    employee_id = possible_id
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

        # ---------------------------------------------------------
        # If a handoff is required, execute it directly
        # ---------------------------------------------------------

        if forced_agent == "call_employee_agent":

            agent_name = "employee_agent"

            print(
                f"\n========== SPECIALIST HANDOFF ==========\n"
            )

            print(
                f"Coordinator is handing the request to "
                f"Employee Agent using the previous specialist's result."
            )

            if agent_name in called_specialists:
                print(
                    "\nWARNING: Employee Agent has already been called."
                )

                break

            called_specialists.add(agent_name)

            print("\n========== SPECIALIST CALL "
                  f"{len(called_specialists)} ==========\n")

            print(
                "Agent: call_employee_agent"
            )

            print(
                f"Arguments: {{'user_query': '{forced_query}'}}"
            )

            specialist_result = call_employee_agent(
                user_query=forced_query
            )

            print("Result:")
            print(
                json.dumps(
                    specialist_result,
                    indent=2
                )
            )

            trace_entry = {
                "agent": "call_employee_agent",
                "arguments": {
                    "user_query": forced_query
                },
                "result": specialist_result
            }

            trace.append(trace_entry)

        else:

            # -----------------------------------------------------
            # Normal Coordinator tool selection
            # -----------------------------------------------------

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
{json.dumps(memory_context, indent=2)}

Execution Trace:
{json.dumps(trace, indent=2)}

Choose the appropriate specialist agent.

Available specialist agents:

1. call_project_agent
   - Project information
   - Project status
   - Project budget
   - Project deadlines
   - Project manager ID

2. call_task_agent
   - Tasks
   - Task status
   - Task assignments
   - Task due dates

3. call_employee_agent
   - Employee information
   - Employee name
   - Employee department
   - Employee role

Call ONLY the specialist required for the current request.

If the previous specialist result already provides the requested
information, do not call another specialist.
"""
                    }
                ],
                tools=list(coordinator_tools.values())
            )

            # -----------------------------------------------------
            # Handle tool call
            # -----------------------------------------------------

            if not response.message.tool_calls:

                # No specialist call was requested.
                # The Coordinator may already have enough information.
                break

            tool_call = response.message.tool_calls[0]

            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            # -----------------------------------------------------
            # Duplicate protection
            # -----------------------------------------------------

            call_signature = (
                tool_name,
                json.dumps(
                    arguments,
                    sort_keys=True
                )
            )

            if call_signature in called_specialists:

                print(
                    "\nWARNING: DUPLICATE SPECIALIST CALL\n"
                )

                print(
                    f"Agent: {tool_name}"
                )

                print(
                    f"Arguments: {arguments}"
                )

                # Do not repeatedly call the same specialist.
                break

            called_specialists.add(call_signature)

            print(
                f"\n========== SPECIALIST CALL "
                f"{len(called_specialists)} ==========\n"
            )

            print(
                f"Agent: {tool_name}"
            )

            print(
                f"Arguments: {arguments}"
            )

            # -----------------------------------------------------
            # Execute specialist
            # -----------------------------------------------------

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

            print("Result:")
            print(
                json.dumps(
                    specialist_result,
                    indent=2
                )
            )

            trace_entry = {
                "agent": tool_name,
                "arguments": arguments,
                "result": specialist_result
            }

            trace.append(trace_entry)

        # ---------------------------------------------------------
        # Observe specialist result
        # ---------------------------------------------------------

        observation = observe_coordinator_result(
            user_query=user_query,
            current_plan=current_plan,
            memory_context=memory_context,
            trace=trace
        )

        print(
            "\n========== COORDINATOR OBSERVATION ==========\n"
        )

        print(
            json.dumps(
                observation,
                indent=2
            )
        )

        # Store observation with the latest trace entry.
        if trace:
            trace[-1]["coordinator_observation"] = observation

        decision = observation.get("decision")

        # ---------------------------------------------------------
        # FINISH
        # ---------------------------------------------------------

        if decision == "FINISH":

            final_answer = create_coordinator_final_answer(
                user_query=user_query,
                memory_context=memory_context,
                trace=trace
            )

            print(
                "\nThe final report/result\n"
            )

            print(
                "=" * 100
            )

            print(final_answer)

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

        # ---------------------------------------------------------
        # CLARIFY
        # ---------------------------------------------------------

        if decision == "CLARIFY":

            clarification = observation.get(
                "reason",
                "Could you please clarify your request?"
            )

            print(
                "\nThe Coordinator needs clarification:\n"
            )

            print(clarification)

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

        # ---------------------------------------------------------
        # REPLAN
        # ---------------------------------------------------------

        if decision == "REPLAN":

            revised_plan = observation.get(
                "revised_plan",
                ""
            )

            if revised_plan:
                current_plan = revised_plan

            else:
                current_plan = plan

            continue

        # ---------------------------------------------------------
        # CONTINUE
        # ---------------------------------------------------------

        if decision == "CONTINUE":

            revised_plan = observation.get(
                "revised_plan",
                ""
            )

            if revised_plan:
                current_plan = revised_plan

            continue

        # ---------------------------------------------------------
        # Unknown decision
        # ---------------------------------------------------------

        print(
            "\nWARNING: Unknown Coordinator decision."
        )

        break

    # -------------------------------------------------------------
    # MAX ITERATIONS / UNRESOLVED REQUEST
    # -------------------------------------------------------------

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