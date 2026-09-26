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
3. Which specialist agent or agents are required.
4. Whether multiple specialist agents are required.
5. Whether one agent's result will be needed by another agent.
6. Whether clarification is required.

IMPORTANT SPECIALIST ROUTING RULE:

- Each specialist is responsible for resolving information within its
  own domain.
- Project Agent is responsible for resolving project names to valid
  project IDs.
- If the user refers to a project by name and another specialist requires
  the project ID, the Project Agent must be used to resolve the project
  name first.
- Do NOT use Task Agent or Employee Agent to resolve a project name.
- Task Agent and Employee Agent should use a valid project ID when their
  requested operation requires one.
- If the Project Agent provides a project ID, reuse that ID when calling
  another specialist.
- Never invent, infer, or construct an identifier when it has not been
  provided or resolved by an appropriate specialist.
- This rule defines responsibility for information resolution; it does
  NOT impose a fixed execution order.
- The overall workflow must still be determined dynamically from the
  user's request, available information, specialist capabilities, and
  results obtained during execution.

Do not invent information.
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

CURRENT COORDINATOR PLAN:
{current_plan}

MEMORY CONTEXT:
{json.dumps(memory_context, indent=2, default=str)}

SPECIALIST EXECUTION TRACE:
{json.dumps(trace, indent=2, default=str)}

Your task is to determine whether the Coordinator should FINISH,
CONTINUE, CLARIFY, or REPLAN.

Use ONLY the ORIGINAL USER REQUEST to determine what information
the user requested.

DECISION RULES:

1. FINISH
Return FINISH when the COMPLETE ORIGINAL USER REQUEST has been satisfied
by the combined specialist results.

2. CONTINUE
Return CONTINUE when:
- the request is clear,
- the specialist results provide only part of the requested information,
  and
- another available specialist can provide the missing information.

If a previous specialist result contains an identifier or other useful
information needed by another specialist, recognize that as a valid
dependency for the next step.

Example:
If the user asks for project status AND in-progress tasks, and the
Project Agent successfully resolves the project to project_id P006,
but the tasks have not yet been retrieved, return CONTINUE.

3. CLARIFY
Return CLARIFY only when the ORIGINAL USER REQUEST itself is genuinely
ambiguous and the Coordinator cannot determine what the user is asking.

Missing information that can be obtained from another specialist is
NOT a reason to return CLARIFY.

4. REPLAN
Return REPLAN only when the current execution approach is no longer
appropriate, such as when a required tool or specialist cannot
reasonably complete the requested objective.

IMPORTANT RULES:

- Do not invent additional user requirements.
- Do not treat a successful partial specialist result as a complete answer
  when another requested part is still missing.
- Do not return FINISH merely because one specialist completed successfully.
- Consider ALL specialist results together before deciding whether the
  ORIGINAL USER REQUEST is complete.
- If multiple specialist results collectively provide all information
  requested by the user, return FINISH.
- Do not request a specialist that is unnecessary.
- Do not repeat an identical specialist call.
- If another specialist is required because a specific part of the
  ORIGINAL USER REQUEST is still missing, return CONTINUE.
- Do not return CONTINUE merely because the original plan contains another
  step. Return CONTINUE only when some requested information is actually
  still missing.
- A valid empty result can satisfy a request if it correctly answers what
  the user asked for.
- Keep the reason concise and factual.
- revised_plan should describe what information still needs to be obtained
  or how the Coordinator should proceed next.
- Return ONLY valid JSON.
- Do NOT use Markdown.
- Do NOT include any text before or after the JSON.
- Use exactly these three keys:
  "decision", "reason", "revised_plan".

VALID OUTPUT FORMAT:

{{
    "decision": "FINISH",
    "reason": "All information required by the original request has been obtained.",
    "revised_plan": ""
}}

OR

{{
    "decision": "CONTINUE",
    "reason": "The project information was obtained, but the requested task information is still missing.",
    "revised_plan": "Use the project identifier discovered by the Project Agent to retrieve the requested task information."
}}

OR

{{
    "decision": "CLARIFY",
    "reason": "The original request is genuinely ambiguous.",
    "revised_plan": ""
}}

OR

{{
    "decision": "REPLAN",
    "reason": "The current execution approach cannot complete the requested objective.",
    "revised_plan": "Create a revised execution approach using the available specialists."
}}

Return ONLY the JSON object.
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
    5. Combines the results into a final answer.
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
{json.dumps(memory_context, indent=2, default=str)}

Execution Trace:
{json.dumps(trace, indent=2, default=str)}

Your job is to choose the NEXT specialist action required to complete
the ORIGINAL USER REQUEST.

Available specialist agents:

1. call_project_agent
   - Project information
   - Project status
   - Project budget
   - Project deadlines
   - Project manager information
   - Project updates
   - Project metrics

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

IMPORTANT EXECUTION RULES:

1. Use exactly ONE specialist call at a time.

2. First inspect the ORIGINAL USER REQUEST, CURRENT COORDINATOR PLAN,
   and EXECUTION TRACE before selecting the next specialist.

3. The EXECUTION TRACE contains results from specialists that have
   already been called. Treat those results as available information.

4. If a previous specialist discovered an identifier or other information
   required by another specialist, use that information when constructing
   the next specialist request.

   Examples:
   - If a Project Agent resolves "Project Zeta" to project_id "P006",
     and tasks for that project are still required, pass "P006" to the
     Task Agent through its user_query.
   - If a Project Agent returns manager_id "E006" and employee details
     are required, pass "E006" to the Employee Agent through its
     user_query.

5. PROJECT NAME RESOLUTION:
   - If the ORIGINAL USER REQUEST refers to a project by project name,
     use the Project Agent to resolve that project name to a valid
     project_id.
   - Do NOT ask the Task Agent or Employee Agent to resolve a project
     name into a project_id.
   - Task Agent and Employee Agent should receive a valid project_id
     when their requested operation requires a project identifier.
   - Once the Project Agent has resolved the project name, reuse the
     discovered project_id when calling another specialist.
   - Never guess or construct a project_id from a project name.

6. Do NOT ask a specialist to rediscover information that has already
   been successfully obtained by a previous specialist.

7. Do NOT call another specialist if the previous specialist results
   already contain everything required by the ORIGINAL USER REQUEST.

8. If the original request requires information from multiple domains,
   continue with the next required specialist after the previous
   specialist has completed.

9. When calling a specialist, make the user_query specific enough for
   that specialist to perform its task using information already
   discovered by previous specialists.

10. Never invent identifiers or other information. Only use identifiers
    that appear in the memory context, coordinator plan, or specialist
    execution trace.

11. Do not repeat an identical specialist call.

12. Do not repeatedly call a specialist merely because it was used
    earlier. A specialist may be called again only when a genuinely
    different request is required and the new call is necessary.

13. The workflow must be dynamically determined from the user's request
    and the information discovered during execution.

14. Do NOT assume a fixed order such as:
    Project Agent -> Task Agent -> Employee Agent.
    The required order depends on the current request and discovered
    information.

For the current execution, choose the single next specialist call that
moves the Coordinator closest to completing the ORIGINAL USER REQUEST.
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