import uuid
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from agents import triage_issue
from database import init_db, save_issue, get_issues, update_review, issue_id_exists

st.set_page_config(
    page_title="Autonomous Bug Triage Agent",
    page_icon="🐞",
    layout="wide"
)
init_db()

st.markdown("""
<style>
.stApp { background-color: #0b1220; }
[data-testid="stMetric"] {
    background: #141f33;
    border: 1px solid #283a56;
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🐞 Autonomous Bug Triage Agent")
st.caption("Generative AI + Agentic AI | Issue classification, duplicate detection and assignment")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Submit Bug", "CSV Batch Triage", "Review Queue", "Issue Explorer"]
)
issues = get_issues()
df = pd.DataFrame(issues)

def run_triage(issue_input, threshold=0.55):
    if issue_id_exists(issue_input["issue_id"]):
        raise ValueError(f"Issue ID {issue_input['issue_id']} already exists.")
    result = triage_issue(issue_input, get_issues(), threshold)
    record = {**issue_input, **result, "result_json": result}
    save_issue(record)
    return result

if page == "Dashboard":
    st.subheader("Engineering Team Overview")
    total = len(df)
    open_count = int((df["status"] == "Open").sum()) if total else 0
    urgent_count = int(df["severity"].isin(["Critical", "High"]).sum()) if total else 0
    duplicate_count = int(df["duplicate_of"].notna().sum()) if total else 0
    pending_count = int((df["review_status"] == "Pending").sum()) if total else 0

    cols = st.columns(5)
    for col, label, value in zip(
        cols,
        ["Total Issues", "Open Issues", "High / Critical", "Potential Duplicates", "Awaiting Review"],
        [total, open_count, urgent_count, duplicate_count, pending_count]
    ):
        col.metric(label, value)

    if df.empty:
        st.info("No issues yet. Submit a bug or upload a CSV to begin.")
    else:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.pie(df, names="severity", title="Severity Distribution", hole=0.4),
                use_container_width=True
            )
        with right:
            priority_order = ["P1", "P2", "P3", "P4"]
            priority_counts = (
                df["priority"].value_counts().reindex(priority_order, fill_value=0)
                .rename_axis("priority").reset_index(name="issues")
            )
            st.plotly_chart(
                px.bar(priority_counts, x="priority", y="issues", title="Issues by Priority"),
                use_container_width=True
            )
        st.plotly_chart(
            px.histogram(df, x="component", color="severity",
                         title="Issues by Component", barmode="stack"),
            use_container_width=True
        )
        assigned = (
            df[df["assignee"].notna() & (df["assignee"] != "")]
            .groupby("assignee", as_index=False).size()
            .rename(columns={"size": "issues"})
        )
        if not assigned.empty:
            st.plotly_chart(
                px.bar(assigned, x="assignee", y="issues", title="Issues by Suggested Developer"),
                use_container_width=True
            )
        st.subheader("Recent Issues")
        show_cols = ["issue_id", "title", "component", "category", "severity",
                     "priority", "assignee", "duplicate_of", "review_status"]
        st.dataframe(df[show_cols].head(10), use_container_width=True, hide_index=True)

elif page == "Submit Bug":
    st.subheader("Submit a New Bug Report")
    with st.form("bug_form"):
        title = st.text_input("Bug title")
        description = st.text_area("Description / error message")
        component = st.selectbox(
            "Component",
            ["Frontend", "API", "Backend", "Database", "Authentication", "Security", "Other"]
        )
        reporter = st.text_input("Reporter", value="Demo User")
        threshold = st.slider(
            "Duplicate similarity threshold", 0.30, 0.95, 0.55, 0.05
        )
        submitted = st.form_submit_button("Run Agentic Triage", type="primary")

    if submitted:
        if not title.strip() or not description.strip():
            st.error("Please provide both a title and description.")
        else:
            issue_input = {
                "issue_id": "BUG-" + uuid.uuid4().hex[:8].upper(),
                "title": title.strip(),
                "description": description.strip(),
                "component": component,
                "reporter": reporter.strip() or "Unknown",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                with st.spinner("Agents are processing the issue..."):
                    result = run_triage(issue_input, threshold)
                st.success(f"Triage completed: {issue_input['issue_id']}")
                a, b, c = st.columns(3)
                a.metric("Category", result["category"])
                b.metric("Severity", result["severity"])
                c.metric("Priority", result["priority"])
                st.write("**Suggested owner:**", result["assignee"])
                st.write("**Classification confidence:**",
                         f"{result['classification_confidence']:.0%}")
                st.write("**Explanation:**", result["explanation"])
                if result["duplicate_of"]:
                    st.warning(
                        f"Possible duplicate of {result['duplicate_of']} "
                        f"(similarity {result['similarity']:.0%}). Review before action."
                    )
                else:
                    st.info("No duplicate was found above the selected threshold.")
                st.caption("Recommendations remain pending until reviewed.")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not process issue: {exc}")

elif page == "CSV Batch Triage":
    st.subheader("Upload and Triage a CSV")
    st.write("Required columns: title, description. Optional: issue_id, component, reporter, status.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    threshold = st.slider("Duplicate similarity threshold", 0.30, 0.95, 0.55, 0.05, key="csv_threshold")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded).fillna("")
            batch.columns = [str(c).strip().lower() for c in batch.columns]
            if not {"title", "description"}.issubset(batch.columns):
                st.error("CSV must contain title and description columns.")
            else:
                st.write(f"Preview: {len(batch)} rows")
                st.dataframe(batch.head(10), use_container_width=True)
                if st.button("Process CSV rows", type="primary"):
                    success, skipped, errors = 0, 0, []
                    progress = st.progress(0)
                    for i, row in batch.iterrows():
                        issue_id = str(row.get("issue_id", "")).strip()
                        if not issue_id or issue_id.lower() == "nan":
                            issue_id = "CSV-" + uuid.uuid4().hex[:8].upper()
                        if issue_id_exists(issue_id):
                            skipped += 1
                            errors.append(f"{issue_id}: already exists")
                            progress.progress((i + 1) / len(batch))
                            continue
                        issue_input = {
                            "issue_id": issue_id,
                            "title": str(row.get("title", "")).strip(),
                            "description": str(row.get("description", "")).strip(),
                            "component": str(row.get("component", "Other")).strip() or "Other",
                            "reporter": str(row.get("reporter", "CSV Import")).strip() or "CSV Import",
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                        try:
                            run_triage(issue_input, threshold)
                            success += 1
                        except Exception as exc:
                            errors.append(f"{issue_id}: {exc}")
                        progress.progress((i + 1) / len(batch))
                    st.success(f"Processed: {success}; skipped: {skipped}; errors: {len(errors)}")
                    if errors:
                        with st.expander("Import details"):
                            st.write("\n".join(errors[:50]))
                    st.rerun()
        except Exception as exc:
            st.error(f"Unable to read CSV: {exc}")

elif page == "Review Queue":
    st.subheader("Human Review and Approval")
    pending = [row for row in issues if row.get("review_status") == "Pending"]
    if not pending:
        st.info("No issues are awaiting review.")
    else:
        options = {f"{row['issue_id']} — {row['title']}": row for row in pending}
        selected = st.selectbox("Choose an issue", list(options.keys()))
        row = options[selected]
        st.write("**Description:**", row["description"])
        st.write("**Category:**", row.get("category"))
        st.write("**Explanation:**", row.get("explanation"))
        if row.get("duplicate_of"):
            st.warning(f"Possible duplicate: {row['duplicate_of']} ({row.get('similarity', 0):.0%})")

        team = ["Asha", "Rahul", "Priya", "Kiran", "Unassigned"]
        severities = ["Critical", "High", "Medium", "Low"]
        priorities = ["P1", "P2", "P3", "P4"]
        with st.form("review_form"):
            old_owner = row.get("assignee")
            owner_index = team.index(old_owner) if old_owner in team else 4
            owner = st.selectbox("Confirm or change owner", team, index=owner_index)
            old_severity = row.get("severity")
            severity = st.selectbox(
                "Severity", severities,
                index=severities.index(old_severity) if old_severity in severities else 2
            )
            old_priority = row.get("priority")
            priority = st.selectbox(
                "Priority", priorities,
                index=priorities.index(old_priority) if old_priority in priorities else 2
            )
            decision = st.radio("Decision", ["Approved", "Rejected"], horizontal=True)
            confirm = st.form_submit_button("Save Review")
        if confirm:
            update_review(
                row["issue_id"], decision,
                None if owner == "Unassigned" else owner,
                severity, priority
            )
            st.success(f"{row['issue_id']} marked {decision}.")
            st.rerun()

elif page == "Issue Explorer":
    st.subheader("Search, Filter and Export")
    if df.empty:
        st.info("No issues available yet.")
    else:
        query = st.text_input("Search issue ID, title, or description")
        severity_filter = st.multiselect(
            "Severity", ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium", "Low"]
        )
        view = df[df["severity"].isin(severity_filter)].copy()
        if query.strip():
            mask = (
                view["issue_id"].fillna("").astype(str).str.contains(query, case=False, regex=False)
                | view["title"].fillna("").astype(str).str.contains(query, case=False, regex=False)
                | view["description"].fillna("").astype(str).str.contains(query, case=False, regex=False)
            )
            view = view[mask]
        columns = [
            "issue_id", "title", "description", "component", "category",
            "severity", "priority", "assignee", "duplicate_of",
            "similarity", "review_status"
        ]
        st.dataframe(view[columns], use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered CSV",
            data=view[columns].to_csv(index=False).encode("utf-8"),
            file_name="bug_triage_results.csv",
            mime="text/csv"
        )

st.sidebar.divider()
st.sidebar.caption("Demo project • Recommendations require human review.")
