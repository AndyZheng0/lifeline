"""Streamlit views for the on-demand account assistant."""
import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import dotenv_values
from streamlit.components.v1 import html
from streamlit.errors import StreamlitSecretNotFoundError

from account_models import ModelService, SCENARIO_FIELDS, ServiceError, parse_action, profile_for


def services():
    local_env = dotenv_values(Path(__file__).with_name(".env"), interpolate=False)

    def configured(group, name, prefix, **paths):
        try:
            settings = st.secrets.get(group, {})
        except StreamlitSecretNotFoundError:
            settings = {}

        def value(field):
            name = f"{prefix}_{field.upper()}"
            return (os.getenv(name) or local_env.get(name) or settings.get(field, "")).strip()

        return ModelService(
            name=name,
            url=value("url"),
            token=value("api_key"),
            model=value("model"),
            **paths,
        )

    decision = configured("decision", "Unsloth Decision", "CHURN", models_path="/v1/models")
    provider = (os.getenv("CHAT_PROVIDER") or local_env.get("CHAT_PROVIDER") or "openwebui").strip().lower()
    if provider == "unsloth":
        speaker = configured(
            "unsloth_chat", "Unsloth Chat", "UNSLOTH_CHAT",
            chat_path="/v1/chat/completions", models_path="/v1/models",
        )
    else:
        speaker = configured("openwebui", "Open WebUI", "OPENWEBUI")
    return decision, speaker


def show_model_settings():
    st.subheader("Account assistant")
    for service in services():
        st.write(f"**{service.name}** · {'Configured' if service.ready() else 'Setup needed'}")
        if st.button(f"List {service.name} model IDs", disabled=not (service.url and service.token)):
            try:
                ids = service.model_ids()
                if ids:
                    st.code("\n".join(ids), language=None)
                else:
                    st.info("The service returned no model IDs for this key.")
            except ServiceError as exc:
                st.error(str(exc))
    st.caption("Enter the API keys, base URLs and model IDs in the .env file beside app.py. "
               "The Streamlit server needs "
               "network access to both model servers.")


def label(result):
    return "Churn" if result["label"] == "churn" else "No churn"


def facts_for(profile, baseline, scenario, change):
    if scenario is None:
        score = f"{baseline['score']:.1%}"
        decision = label(baseline).lower()
        return [
            f"The Decision Model predicts {decision} for this account and returned a churn score of {score}.",
            f"For the selected account, the model returned a churn score of {score} and a {decision} label.",
        ]
    field, value = change
    delta = 100 * (scenario["score"] - baseline["score"])
    field_name = SCENARIO_FIELDS[field].lower()
    direction = "left the score nearly unchanged" if abs(delta) < .005 else (
        "lowered the score" if delta < 0 else "raised the score"
    )
    return [
        f"In this test, changing {field_name} from {profile[field]} to {value} {direction}.",
        f"The comparison {direction} when I changed {field_name} from {profile[field]} to {value}.",
    ]


def explain_with_chat(speaker, question, facts):
    if speaker.ready() and len(facts) > 1:
        try:
            answer = speaker.chat(
                "Choose the supplied observation that best answers the analyst. "
                "Reply with that exact sentence, without quotes or extra claims.",
                {"question": question, "observations": facts},
            )
            if answer in facts:
                return answer
        except ServiceError:
            pass
    return facts[0]


def handle_question(question, profile, thread, decision, speaker, direct=False):
    if direct:
        action, change = "score", None
    else:
        history = [item["question"] for item in thread["turns"]]
        action, change = parse_action(question, profile, history, speaker)
    if action == "clarify":
        return {"text": "What single change should I test? Try “What if they had two fewer complaints?”"}
    if action == "chat":
        answer = speaker.chat(
            "You are LifeLine’s telecom retention assistant. Help the analyst understand "
            "the selected customer profile and the dashboard. Reply briefly in plain text. "
            "Use the provided profile for account facts. Only the Decision API can predict churn. "
            "Do not invent scores, explanations, real-world causes, or model accuracy. "
            "If asked for an unsupported action, explain the available account scoring and scenario features.",
            {"question": question, "profile": profile,
             "previous_questions": [item["question"] for item in thread["turns"][-3:]]},
        )
        return {"text": answer}

    if action == "scenario" and profile[change[0]] == change[1]:
        return {"text": "That value already matches the selected account. Give me a different value to compare."}
    if thread["baseline"] is None:
        thread["baseline"] = decision.score(profile)
    baseline = thread["baseline"]
    scenario = None
    note = ""
    if action == "explain":
        complaints = profile["total_complaints"]
        change = ("total_complaints", max(0, complaints - 2) if complaints else 2)
    if change is not None:
        field, value = change
        try:
            scenario = decision.score({**profile, field: value})
        except ServiceError as exc:
            note = f"The original score is available. The comparison failed. {exc}"
    facts = facts_for(profile, baseline, scenario, change)
    return {"text": explain_with_chat(speaker, question, facts), "baseline": baseline,
            "scenario": scenario, "note": note}


def read_aloud(text):
    spoken = json.dumps(text).replace("<", "\\u003c")
    html(f"""
        <button type="button" id="speak" hidden>Read aloud</button>
        <script>
        const button = document.getElementById('speak');
        if ('speechSynthesis' in window) {{
            button.hidden = false;
            button.onclick = () => {{
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(new SpeechSynthesisUtterance({spoken}));
            }};
        }}
        </script>
    """, height=36)


def show_turn(turn, latest=False):
    with st.chat_message("user"):
        st.text(turn["question"])
    with st.chat_message("assistant"):
        st.text(turn["text"])
        spoken = turn["text"]
        if "baseline" in turn:
            base = turn["baseline"]
            scenario = turn["scenario"]
            scores = {"Original": base["score"]}
            if scenario is None:
                st.metric(f"Decision Model · {label(base)}", f"{base['score']:.1%}")
            else:
                original, changed = st.columns(2)
                original.metric(f"Original · {label(base)}", f"{base['score']:.1%}")
                changed.metric(f"Scenario · {label(scenario)}", f"{scenario['score']:.1%}",
                               f"{100 * (scenario['score'] - base['score']):+.2f} pp",
                               delta_color="off")
                scores["Scenario"] = scenario["score"]
                st.bar_chart(pd.Series(scores, name="Churn score"), horizontal=True)
                spoken += f" The original churn score was {base['score']:.1%}. The scenario score was {scenario['score']:.1%}."
            st.caption("The score comes from the Decision Model. What-if comparisons measure how its score responds to a change.")
            if turn["note"]:
                st.warning(turn["note"])
        if latest:
            read_aloud(spoken)


def show_account_chat(customers):
    st.subheader("Account assistant")
    st.caption("Select an account, then ask for a churn prediction, an explanation, or a what-if comparison.")
    account_id = st.selectbox("Account to analyze", customers["CustomerID"].tolist(), index=None,
                              placeholder="Select a customer ID", key="assistant_customer")
    if account_id is None:
        return
    row = customers.loc[customers["CustomerID"].eq(account_id)].iloc[0]
    try:
        profile = profile_for(row)
    except ValueError as exc:
        st.warning(str(exc))
        return
    decision, speaker = services()
    if not decision.ready() or not speaker.ready():
        st.info("Check Model + Settings to configure the decision and chat connections.")
    threads = st.session_state.setdefault("account_chat_threads", {})
    thread = threads.get(account_id)
    decision_identity = (decision.url, decision.model)
    if thread is None or thread["profile"] != profile or thread.get("decision_identity") != decision_identity:
        thread = threads[account_id] = {
            "profile": profile, "turns": [], "baseline": None,
            "decision_identity": decision_identity,
        }
    for index, turn in enumerate(thread["turns"]):
        show_turn(turn, latest=index == len(thread["turns"]) - 1)
    controls, reset = st.columns([3, 1])
    score_clicked = controls.button("Predict churn", disabled=not decision.ready())
    if reset.button("Clear chat", disabled=not thread["turns"]):
        del threads[account_id]
        st.rerun()
    with st.container():
        question = st.chat_input("Ask about this account…", max_chars=1200, disabled=not speaker.ready())
    if score_clicked or question:
        question = "Predict churn for this account." if score_clicked else question
        try:
            with st.spinner("Reviewing the account…"):
                reply = handle_question(question, profile, thread, decision, speaker, direct=score_clicked)
        except ServiceError as exc:
            reply = {"text": str(exc)}
        turn = {"question": question, **reply}
        thread["turns"].append(turn)
        st.rerun()
