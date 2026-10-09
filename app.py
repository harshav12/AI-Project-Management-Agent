import streamlit as st
from my_agent.Coordinator_agent import run_coordinator

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Project Management Agent",
    page_icon="🤖",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.25);
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .main-subtitle {
        color: #777;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .online-status {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 12px;
        border-radius: 20px;
        background: rgba(40, 167, 69, 0.10);
        color: #28a745;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }

    .online-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #28a745;
    }

    .user-card {
        padding: 14px;
        border-radius: 10px;
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .user-name {
        font-weight: 600;
        font-size: 1rem;
    }

    .user-role {
        font-size: 0.82rem;
        color: #777;
        margin-top: 3px;
    }

    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #777;
        margin-bottom: 8px;
    }

    .info-note {
        color: #777;
        font-size: 0.82rem;
        margin-top: 4px;
        margin-bottom: 0;
    }

    .permission-allowed {
        color: #28a745;
        font-weight: 600;
    }

    .permission-denied {
        color: #999;
        font-weight: 500;
    }

    .footer {
        text-align: center;
        color: #888;
        font-size: 0.75rem;
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# USER DATA
# ==================================================

users = {
    "Emily Johnson (E005) - Project Manager": "E005",
    "David Wilson (E006) - Tech Lead": "E006",
    "Ava Thomas (E011) - Product Manager": "E011",
    "Sarah Williams (E002) - Software Engineer": "E002"
}

# ==================================================
# GEMINI MODELS
# ==================================================

gemini_models = [
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite"
]

# ==================================================
# ROLE PERMISSIONS
# ==================================================

# ==================================================
# ROLE PERMISSIONS
# ==================================================

role_permissions = {
    "Project Manager": {
        "View Projects": True,
        "View Tasks": True,
        "Update Tasks": True,
        "View Employees": True,
        "View Project Metrics": True,
        "View Knowledge Base": True
    },
    "Tech Lead": {
        "View Projects": True,
        "View Tasks": True,
        "Update Tasks": True,
        "View Employees": True,
        "View Project Metrics": True,
        "View Knowledge Base": True
    },
    "Product Manager": {
        "View Projects": True,
        "View Tasks": True,
        "Update Tasks": False,
        "View Employees": False,
        "View Project Metrics": True,
        "View Knowledge Base": True
    },
    "Software Engineer": {
        "View Projects": True,
        "View Tasks": True,
        "Update Tasks": False,
        "View Employees": False,
        "View Project Metrics": False,
        "View Knowledge Base": True
    }
}

# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================================================
# EXECUTION DETAILS
# ==================================================

def render_execution_details(
    trace,
    iterations,
    memory_context=None
):
    with st.expander("🔍 Execution Details"):
        st.caption(
            f"Coordinator iterations: {iterations}"
        )

        # --------------------------------------------------
        # MEMORY CONTEXT
        # --------------------------------------------------

        if memory_context:
            st.markdown(
                "🧠 **Memory Context Used**"
            )
            st.caption(
                "Relevant user memory was retrieved for this request."
            )

        if not trace:
            st.caption("No execution trace available.")
        else:
            for step_index, step in enumerate(trace, start=1):
                agent = step.get(
                    "agent",
                    "Unknown Agent"
                )

                agent_result = step.get(
                    "result",
                    {}
                )

                st.markdown(
                    f"**{step_index}. Coordinator → {agent}**"
                )

                if isinstance(agent_result, dict):
                    specialist_agent = agent_result.get("agent")

                    if specialist_agent:
                        st.markdown(
                            f"↳ 🤖 **{specialist_agent}**"
                        )

                    specialist_trace = agent_result.get(
                        "trace",
                        []
                    )

                    for tool_step in specialist_trace:
                        tool_name = tool_step.get(
                            "tool",
                            "Unknown Tool"
                        )

                        tool_arguments = tool_step.get(
                            "arguments",
                            {}
                        )

                        tool_result = tool_step.get(
                            "result",
                            {}
                        )

                        st.markdown(
                            f"↳ 🔧 `{tool_name}`"
                        )

                        if isinstance(tool_arguments, dict):
                            for key, value in tool_arguments.items():
                                st.caption(
                                    f"{key.replace('_', ' ').title()}: {value}"
                                )

                        if isinstance(tool_result, dict):
                            if tool_result.get("success") is False:
                                st.error(
                                    "✗ Tool execution failed"
                                )
                            else:
                                st.success(
                                    "✓ Completed",
                                    icon="✅"
                                )
                        else:
                            st.success(
                                "✓ Completed",
                                icon="✅"
                            )

                if step_index < len(trace):
                    st.divider()

            st.divider()

            # Check whether any specialist tool failed
            has_tool_failure = any(
                isinstance(specialist_step.get("result"), dict)
                and any(
                    isinstance(tool_step.get("result"), dict)
                    and tool_step.get("result", {}).get("success") is False
                    for tool_step in specialist_step.get("result", {}).get("trace", [])
                )
                for specialist_step in trace
            )

            if has_tool_failure:
                st.warning(
                    "Execution completed with tool errors.",
                    icon="⚠️"
                )
            else:
                st.success(
                    "Execution completed successfully.",
                    icon="🤖"
                )

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.title("🤖 Project Agent")

    st.markdown(
        '<div class="section-label">Current User</div>',
        unsafe_allow_html=True
    )

    selected_user = st.selectbox(
        "Select User",
        options=list(users.keys()),
        label_visibility="collapsed"
    )

    user_id = users[selected_user]
    user_name = selected_user.split(" (")[0]
    user_role = selected_user.split(" - ")[1]

    st.markdown(
        f"""
        <div class="user-card">
            <div class="user-name">👤 {user_name}</div>
            <div class="user-role">{user_role}</div>
            <div class="user-role">ID: {user_id}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # ==================================================
    # DOCUMENT UPLOAD
    # ==================================================

    st.markdown("### 📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Select documents to upload",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="document_uploader"
    )

    if uploaded_files:
        st.caption(
            f"{len(uploaded_files)} document(s) selected."
        )

    st.divider()


    # ==================================================
    # GEMINI MODEL SELECTION
    # ==================================================

    st.markdown(
        '<div class="section-label">Gemini Model</div>',
        unsafe_allow_html=True
    )

    selected_model = st.selectbox(
        "Select Gemini Model",
        options=gemini_models,
        index=0,
        label_visibility="collapsed"
    )

    st.caption(
        f"Using: `{selected_model}`"
    )

    st.divider()

    st.markdown(
        '<div class="section-label">Available Users</div>',
        unsafe_allow_html=True
    )

    for user, uid in users.items():
        name = user.split(" (")[0]
        st.caption(f"**{uid}** · {name}")

    st.divider()

    st.markdown(
        """
        <div class="online-status">
            <span class="online-dot"></span>
            System Online
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

# ==================================================
# MAIN HEADER
# ==================================================

st.markdown(
    '<div class="main-title">Project Management Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">Multi-Agent AI Project Management System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="online-status">
        <span class="online-dot"></span>
        Assistant Ready
    </div>
    """,
    unsafe_allow_html=True
)

# ==================================================
# HOW TO USE & PERMISSIONS
# ==================================================

col1, col2 = st.columns(2)

with col1:
    with st.expander("📖 How to use"):
        st.markdown(
            """
            Ask questions in natural language. The system routes your request
            to the appropriate specialist agent.

            **You can ask about:**

            - 📁 **Projects** — status, managers, deadlines, budgets and risks
            - 📋 **Tasks** — status, priorities, assignees, progress and due dates
            - ✏️ **Task Updates** — update task status and assign or reassign tasks
            - 👥 **Employees** — employee information, roles and departments
            - 📊 **Project Metrics** — project-level performance and metrics
            - 📚 **Guidelines & Policies** — company, project and development guidelines

            **Examples:**
            - *What is the current status of Project Alpha?*
            - *What is the most discussed project?*
            - *Who is working on the project P001?*
            - *What are the development guidelines?*
            - *Show me the project metrics.*
            - *Update task T003 status to completed.*
            - *Reassign task T003 to E006.*
            """
        )

with col2:
    with st.expander("🔐 Your permissions"):
        current_permissions = role_permissions.get(
            user_role,
            {}
        )

        st.markdown(
            f"**Current role:** {user_role}"
        )

        for permission, allowed in current_permissions.items():
            if allowed:
                st.markdown(
                    f'<div class="permission-allowed">✓ {permission}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="permission-denied">✗ {permission}</div>',
                    unsafe_allow_html=True
                )

        st.markdown(
            '<p class="info-note">Some requests may be unavailable depending on your role and permissions.</p>',
            unsafe_allow_html=True
        )

# ==================================================
# DISPLAY CONVERSATION HISTORY
# ==================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "trace" in message:
            render_execution_details(
                message["trace"],
                message.get("iterations", 0),
                message.get("memory_context")
            )

# ==================================================
# EMPTY STATE
# ==================================================

if not st.session_state.messages:
    st.info(
        "👋 Ask me about projects, tasks, employees, or project progress."
    )

# ==================================================
# CHAT INPUT
# ==================================================

query = st.chat_input(
    "Ask about projects, tasks, employees..."
)

# ==================================================
# HANDLE NEW MESSAGE
# ==================================================

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # ==================================================
    # RUN REAL COORDINATOR
    # ==================================================

    with st.chat_message("assistant"):
        try:
            with st.spinner("🤖 Coordinator is working..."):
                result = run_coordinator(
                    user_id=user_id,
                    user_query=query,
                    model=selected_model
                )

            # ------------------------------------------
            # VALIDATE RESULT
            # ------------------------------------------

            if not isinstance(result, dict):
                response = str(result)

                st.markdown(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            # ------------------------------------------
            # SUCCESSFUL RESPONSE
            # ------------------------------------------

            elif result.get("success"):
                response = result.get(
                    "report"
                )

                if not response:
                    response = (
                        "The request was completed, "
                        "but no report was returned."
                    )

                trace = result.get(
                    "trace",
                    []
                )

                iterations = result.get(
                    "iterations",
                    0
                )

                memory_context = result.get(
                    "memory_context",
                    {}
                )

                st.markdown(response)

                render_execution_details(
                    trace,
                    iterations,
                    memory_context
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "trace": trace,
                    "iterations": iterations,
                    "memory_context": memory_context
                })

            # ------------------------------------------
            # COORDINATOR RETURNED FAILURE
            # ------------------------------------------

            else:
                response = result.get(
                    "report",
                    "The Coordinator could not complete the request."
                )

                st.error(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

        # ----------------------------------------------
        # BACKEND ERROR
        # ----------------------------------------------

        except Exception as e:
            st.error(
                "The request could not be processed."
            )

            with st.expander("Technical Details"):
                st.code(
                    str(e),
                    language="text"
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "Sorry, I couldn't process that request."
                )
            })

# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">
        Project Management Agent · Multi-Agent AI System
    </div>
    """,
    unsafe_allow_html=True
)