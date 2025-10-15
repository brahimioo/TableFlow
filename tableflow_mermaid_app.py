"""
Streamlit app to visualise call-flows defined in a CSV using Mermaid diagrams.

This app was built specifically to accommodate CSVs similar to the user's
`Voice_StandardFlow09102025.csv`.  The core logic performs the following steps:

1. Load a CSV describing a series of call flow steps.  The expected
   columns include at minimum the following fields:

   - ``key``: the unique identifier of each step.  This value functions as
     the ``FlowKeyword`` used throughout the flow definition.  Keys may
     contain underscores or commas; these characters are preserved in labels
     but sanitised for use in Mermaid node identifiers.
   - ``Action``: the type of step.  Recognised actions include
     ``Menu``, ``Transfer``, ``Schedule``, ``Jump``, ``Callback``,
     ``Emergency`` and ``Disconnect``.  Any other value is treated as a
     terminal action with no outgoing edges.
   - ``TransferTo``: destination for ``Transfer`` actions.  The value
     generally follows the pattern ``Queue:<name>,...``, ``User:<name>``,
     ``Number:<digits>`` or ``Flow:<flowname>``.  Only the first segment
     before a comma is used to resolve a link to another step.  If that
     segment matches another ``key`` in the table, the edge will point to
     that node; otherwise a terminal node is created with the full
     ``TransferTo`` string as its label.
   - ``JumpToLocation``: destination for ``Jump``, ``Callback`` and
     ``Emergency`` actions.  If the value matches a ``key`` in the table
     the edge points to that node; otherwise a terminal node is created
     displaying the target text.
   - ``PossibleMenuOptions`` and ``DefaultMenuOption``: comma separated
     option values for ``Menu`` actions.  For example ``1,88`` in
     ``PossibleMenuOptions`` combined with ``88`` as ``DefaultMenuOption``
     yields two edges labelled ``1`` and ``88``.  Each option value is
     concatenated with the current step key using a comma to form the
     destination key (e.g. ``NL_BSN,1``).  If such a key exists in the
     table it is linked; otherwise a terminal node is created labelled
     with the option value.
   - ``ScheduleGroup``: base name for ``Schedule`` actions.  ``Schedule``
     actions always fan out into three branches labelled ``Open``,
     ``Closed`` and ``Holiday``.  The destination key is formed by
     appending ``_Open``, ``_Closed`` or ``_Holiday`` to the current
     step's ``key``.  If a matching key exists in the table it is linked;
     otherwise a terminal node is created with the status as its label.

This script performs minimal validation beyond ensuring the required
columns are present.  When run via ``streamlit``, the GUI allows you to
upload a CSV, edit it in a table, select a root flow to display and
export your changes back to CSV.  Mermaid diagrams are rendered in the
browser via an embedded Mermaid.js runtime.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd

try:
    # Streamlit is only needed when running the application.  Import it lazily
    import streamlit as st  # type: ignore
except ModuleNotFoundError:
    st = None  # type: ignore


def mmd_label(text: Optional[str]) -> str:
    """Sanitise a string for safe use in a Mermaid label.

    Mermaid nodes use square brackets to delimit labels, and double quotes
    within labels can break the parser.  Replace brackets with parentheses
    and double quotes with single quotes.  Collapse whitespace to a single
    space.

    Args:
        text: Original text, possibly ``None``.

    Returns:
        Sanitised label string.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    # Replace bracket-like characters to avoid ending node definitions
    s = s.replace("[", "(").replace("]", ")")
    # Replace double quotes with single quotes
    s = s.replace('"', "'")
    # Collapse whitespace and strip
    s = " ".join(s.split())
    return s


def clean_id(raw: str, taken: Optional[Set[str]] = None) -> str:
    """Generate a valid Mermaid node identifier from arbitrary text.

    Mermaid identifiers must start with a letter or underscore and may
    contain only letters, digits and underscores.  This function replaces
    any invalid character with an underscore and ensures uniqueness if
    ``taken`` is provided.

    Args:
        raw: Arbitrary identifier text.
        taken: Optional set of already-used identifiers.  If provided,
            a numeric suffix will be appended to ensure uniqueness.

    Returns:
        A valid Mermaid identifier string.
    """
    # Replace all non-word characters with underscores
    cleaned = re.sub(r"\W", "_", raw)
    # Mermaid identifiers must not start with a digit
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    if taken is not None:
        base = cleaned
        counter = 1
        while cleaned in taken:
            cleaned = f"{base}_{counter}"
            counter += 1
        taken.add(cleaned)
    return cleaned


def build_mermaid(df: pd.DataFrame, root: str) -> str:
    """Construct a Mermaid flowchart (top-down) for a given root key.

    The function walks the flow defined in ``df`` starting from ``root``
    and recursively emits Mermaid node and edge definitions.  Unknown
    destinations produce terminal nodes.  Node identifiers are
    sanitised to avoid conflicts with Mermaid syntax.

    Args:
        df: DataFrame containing at least the columns ``key``, ``Action``,
            ``TransferTo``, ``JumpToLocation``, ``PossibleMenuOptions``,
            ``DefaultMenuOption`` and ``ScheduleGroup``.
        root: The root ``key`` value to build the flow for.

    Returns:
        A string containing the complete Mermaid chart, including class
        definitions to colour nodes by action type.
    """
    # Build a lookup dict from key to row and a lower-case index for case-insensitive lookup
    lookup: Dict[str, pd.Series] = {str(row['key']): row for _, row in df.iterrows()}
    # Map lowercase keys to the canonical key for case-insensitive resolution
    lower_map: Dict[str, str] = {str(k).lower(): str(k) for k in lookup.keys()}

    # Keep track of node identifiers we've used
    taken: Set[str] = set()
    # Map original keys to sanitized IDs to avoid duplicates across recursive calls
    id_map: Dict[str, str] = {}
    # List to accumulate node definitions and edge definitions
    node_lines: List[str] = []
    edge_lines: List[str] = []

    def get_id(key: str, label: str, action_type: str) -> str:
        """Return a unique Mermaid identifier for a given key.

        If the key has already been assigned an identifier, reuse it; otherwise
        generate a new one.  Side effect: records the node definition.
        """
        # If we've already created a node for this key, return its id
        if key in id_map:
            return id_map[key]
        nid = clean_id(key, taken)
        id_map[key] = nid
        # Emit the node with label and class based on action type
        node_lines.append(f"{nid}[{mmd_label(label)}]:::action_{action_type.upper()}")
        return nid

    def add_terminal(parent_key: str, label: str, action_type: str) -> str:
        """Create a terminal node for edges that have no explicit target.

        Terminal nodes are uniquely identified by combining the parent key
        with the label, then sanitised.  Their label includes the action
        type for clarity.
        """
        # Compose a unique raw id from parent key and label
        raw = f"{parent_key}_{label}"
        nid = clean_id(raw, taken)
        node_lines.append(
            f"{nid}[{mmd_label(label)}]:::action_{action_type.upper()}"
        )
        return nid

    # Track visited keys to avoid infinite loops
    visited: Set[str] = set()

    # Lowercase representation of the currently selected root flow.  Used to
    # determine whether a Flow: transfer points back into the same flow or
    # to a completely separate flow.  External flows are shown as single
    # terminal nodes rather than expanding the entire flow.
    root_lower = root.lower()

    def walk(key: str) -> str:
        """Recursively traverse the flow starting from ``key``.

        Returns the Mermaid node identifier for the starting step.
        """
        # If we've already walked this key, return its node id
        if key in visited:
            return id_map.get(key, clean_id(key, taken))
        visited.add(key)
        # Retrieve the row; if absent, create a generic terminal node
        # Resolve key case-insensitively
        row_key = key
        if key not in lookup:
            lk = key.lower()
            # Map to canonical if exists
            if lk in lower_map:
                row_key = lower_map[lk]
            else:
                row_key = key
        row = lookup.get(row_key)
        if row is None:
            return add_terminal(key, key, "Unknown")
        action = str(row['Action']) if not pd.isna(row['Action']) else ""
        act_upper = action.upper()
        # Compose a label with action and key for clarity
        label = f"{act_upper}: {key}" if action else key
        # Determine the class of this action: map unrecognised actions to UNKNOWN
        recognised = {
            'MENU', 'TRANSFER', 'SCHEDULE', 'JUMP', 'CALLBACK', 'EMERGENCY', 'DISCONNECT'
        }
        action_class = act_upper if act_upper in recognised else 'UNKNOWN'
        current_id = get_id(key, label, action_class)
        # Dispatch based on action type
        if act_upper == 'MENU':
            # Parse options and default
            opt_str = row.get('PossibleMenuOptions')
            default_opt = row.get('DefaultMenuOption')
            options: List[str] = []
            if pd.notna(opt_str):
                # Split on commas and strip whitespace
                options = [o.strip() for o in str(opt_str).split(',') if o.strip()]
            # Append the default as a separate edge if present and not already in options
            if pd.notna(default_opt):
                def_opt = str(default_opt).strip()
                # Prepend default marker '*' to indicate default option in label
                if def_opt not in options:
                    options.append(def_opt)
            for opt in options:
                # Derive the target key: current key plus comma and option
                target_key = f"{key},{opt}"
                # Determine label: mark default differently
                edge_label = opt
                # If the default option is represented by '*', clarify label
                if opt == '*' or opt == '✱':
                    edge_label = 'default'
                # Check if such a step exists; if so, recurse
                # resolve case-insensitive
                # If such a step exists; if so, recurse
                tgt_key = target_key
                if target_key not in lookup:
                    lk = target_key.lower()
                    if lk in lower_map:
                        tgt_key = lower_map[lk]
                if tgt_key in lookup:
                    tgt_id = walk(tgt_key)
                else:
                    # Terminal node labelled by the option
                    tgt_id = add_terminal(key, f"opt {opt}", 'MENU')
                edge_lines.append(f"{current_id} -->|{mmd_label(edge_label)}| {tgt_id}")
        elif act_upper == 'TRANSFER':
            transf = row.get('TransferTo')
            if pd.notna(transf):
                tstr = str(transf).strip()
                # Normalise multiple whitespace
                tstr = " ".join(tstr.split())
                # Parse prefix and target part before the first comma
                prefix_part = tstr.split(',', 1)[0]
                # Extract the destination type and the name
                if ':' in prefix_part:
                    dest_type, dest_value = prefix_part.split(':', 1)
                else:
                    dest_type, dest_value = '', prefix_part
                dest_value = dest_value.strip()
                # If the transfer points to another flow step via Flow: prefix
                if dest_type.upper() == 'FLOW':
                    # FLOW transfers link to other keys if they belong to the same flow.
                    # Otherwise they are represented as a single terminal node showing
                    # only the destination flow name.  This avoids drawing the entire
                    # separate flow under the current flow.
                    dest_key = dest_value
                    # Determine whether the destination flow appears to be the same as
                    # the current root.  Compare case-insensitively.
                    if dest_value.strip().lower() == root_lower:
                        # Same flow: resolve case-insensitive and recurse
                        tgt_key = dest_key
                        if dest_key not in lookup:
                            lk = dest_key.lower()
                            if lk in lower_map:
                                tgt_key = lower_map[lk]
                        if tgt_key in lookup:
                            tgt_id = walk(tgt_key)
                        else:
                            # unknown target in same flow – display a terminal
                            tgt_id = add_terminal(key, dest_value, 'TRANSFER')
                    else:
                        # Different flow: show only the entry point of the target flow
                        tgt_id = add_terminal(key, dest_value, 'TRANSFER')
                else:
                    tgt_id = add_terminal(key, tstr, 'TRANSFER')
                edge_lines.append(f"{current_id} -->|transfer| {tgt_id}")
            else:
                # No transfer info – treat as terminal
                term_id = add_terminal(key, 'Transfer', 'TRANSFER')
                edge_lines.append(f"{current_id} -->|transfer| {term_id}")
        elif act_upper == 'JUMP':
            dest = row.get('JumpToLocation')
            if pd.notna(dest):
                dest_key = str(dest).strip()
                # resolve case-insensitive
                tgt_key = dest_key
                if dest_key not in lookup:
                    lk = dest_key.lower()
                    if lk in lower_map:
                        tgt_key = lower_map[lk]
                if tgt_key in lookup:
                    tgt_id = walk(tgt_key)
                else:
                    tgt_id = add_terminal(key, dest_key, 'JUMP')
                edge_lines.append(f"{current_id} -->|jump| {tgt_id}")
            else:
                term_id = add_terminal(key, 'Jump', 'JUMP')
                edge_lines.append(f"{current_id} -->|jump| {term_id}")
        elif act_upper == 'SCHEDULE':
            # For schedule we always branch to Open, Closed and Holiday.  Resolve
            # destination keys case-insensitively.
            for status in ('Open', 'Closed', 'Holiday'):
                dest_key = f"{key}_{status}"
                # Try case-insensitive resolution
                tgt_key = dest_key
                if dest_key not in lookup:
                    lk = dest_key.lower()
                    if lk in lower_map:
                        tgt_key = lower_map[lk]
                if tgt_key in lookup:
                    tgt_id = walk(tgt_key)
                else:
                    tgt_id = add_terminal(key, status, 'SCHEDULE')
                edge_lines.append(f"{current_id} -->|{status.lower()}| {tgt_id}")
        elif act_upper == 'CALLBACK':
            dest = row.get('JumpToLocation')
            if pd.notna(dest):
                dest_key = str(dest).strip()
                tgt_key = dest_key
                if dest_key not in lookup:
                    lk = dest_key.lower()
                    if lk in lower_map:
                        tgt_key = lower_map[lk]
                if tgt_key in lookup:
                    tgt_id = walk(tgt_key)
                else:
                    tgt_id = add_terminal(key, dest_key, 'CALLBACK')
                edge_lines.append(f"{current_id} -->|callback| {tgt_id}")
            else:
                term_id = add_terminal(key, 'Callback', 'CALLBACK')
                edge_lines.append(f"{current_id} -->|callback| {term_id}")
        elif act_upper == 'EMERGENCY':
            dest = row.get('JumpToLocation')
            if pd.notna(dest):
                dest_key = str(dest).strip()
                tgt_key = dest_key
                if dest_key not in lookup:
                    lk = dest_key.lower()
                    if lk in lower_map:
                        tgt_key = lower_map[lk]
                if tgt_key in lookup:
                    tgt_id = walk(tgt_key)
                else:
                    tgt_id = add_terminal(key, dest_key, 'EMERGENCY')
                edge_lines.append(f"{current_id} -->|emergency| {tgt_id}")
            else:
                term_id = add_terminal(key, 'Emergency', 'EMERGENCY')
                edge_lines.append(f"{current_id} -->|emergency| {term_id}")
        elif act_upper == 'DISCONNECT':
            # Represent disconnect as a terminal node
            term_id = add_terminal(key, 'Disconnect', 'DISCONNECT')
            edge_lines.append(f"{current_id} -->|disconnect| {term_id}")
        else:
            # Any unrecognised action type is treated as terminal.  Use
            # the action name for the edge label and assign the node to
            # the generic UNKNOWN class.
            term_id = add_terminal(key, action or 'End', 'UNKNOWN')
            edge_lines.append(f"{current_id} -->|{mmd_label(action)}| {term_id}")
        return current_id

    # Kick off traversal from root
    walk(root)
    # Deduplicate node and edge lines whilst preserving order
    unique_nodes: List[str] = []
    seen_nodes: Set[str] = set()
    for ln in node_lines:
        if ln not in seen_nodes:
            unique_nodes.append(ln)
            seen_nodes.add(ln)
    unique_edges: List[str] = []
    seen_edges: Set[str] = set()
    for ln in edge_lines:
        if ln not in seen_edges:
            unique_edges.append(ln)
            seen_edges.add(ln)
    # Compose class definitions with colours per action
    class_defs = "\n".join(
        [
            "classDef action_MENU fill:#FFD43B,stroke:#333,stroke-width:1px;",
            "classDef action_TRANSFER fill:#4DABF7,stroke:#333,stroke-width:1px;",
            "classDef action_DISCONNECT fill:#FA5252,stroke:#333,stroke-width:1px;",
            "classDef action_JUMP fill:#BE4BDB,stroke:#333,stroke-width:1px;",
            "classDef action_SCHEDULE fill:#51CF66,stroke:#333,stroke-width:1px;",
            "classDef action_CALLBACK fill:#94D82D,stroke:#333,stroke-width:1px;",
            "classDef action_EMERGENCY fill:#FFA94D,stroke:#333,stroke-width:1px;",
            "classDef action_UNKNOWN fill:#ADB5BD,stroke:#333,stroke-width:1px;",
            "classDef action_END fill:#CED4DA,stroke:#333,stroke-width:1px;",
        ]
    )
    # Build the final Mermaid chart
    mermaid = "flowchart TD\n" + "\n".join(unique_nodes + unique_edges) + "\n" + class_defs
    return mermaid


def validate_flows(df: pd.DataFrame) -> List[str]:
    """Validate flow definitions and return a list of warnings.

    The validator looks for common mistakes such as:

    * Schedule actions where expected ``_Open``, ``_Closed`` or ``_Holiday``
      branches are missing or misnamed with a comma instead of an underscore.
    * Menu actions that refer to options for which no destination key exists.
    * Jump, Callback and Emergency actions pointing to non‑existent keys.
    * Transfer actions using ``Flow:`` prefix that point to non‑existent keys.

    Validation is **case‑insensitive**: all keys and targets are compared in
    lowercase to avoid spurious errors when uppercase/lowercase differ.

    Args:
        df: DataFrame containing at least ``key``, ``Action``, ``TransferTo``,
            ``JumpToLocation``, ``PossibleMenuOptions`` and ``DefaultMenuOption``.

    Returns:
        A list of human‑readable warning messages.  An empty list means no
        issues were detected.
    """
    warnings: List[str] = []
    # Build a set of all keys (lowercase) for quick case‑insensitive existence checks
    keys_set: Set[str] = set(str(k).strip().lower() for k in df['key'].dropna())
    for _, row in df.iterrows():
        key = row['key']
        if pd.isna(key):
            continue
        key_str = str(key).strip()
        key_lower = key_str.lower()
        action = row['Action']
        action_str = str(action).strip() if pd.notna(action) else ''
        act_upper = action_str.upper()
        if act_upper == 'SCHEDULE':
            # Schedule actions must have _Open, _Closed and _Holiday suffixes
            for suffix in ('Open', 'Closed', 'Holiday'):
                expected = f"{key_str}_{suffix}"
                comma_variant = f"{key_str},{suffix}"
                # Check existence case‑insensitively
                expected_lower = expected.strip().lower()
                comma_lower = comma_variant.strip().lower()
                if expected_lower not in keys_set:
                    if comma_lower in keys_set:
                        warnings.append(
                            f"{key_str}: schedule branch '{suffix}' uses comma instead of underscore: '{comma_variant}' (expected '{expected}')."
                        )
                    else:
                        warnings.append(
                            f"{key_str}: missing schedule branch '{suffix}'. Expected key '{expected}'."
                        )
        elif act_upper == 'MENU':
            # Menu actions must have keys for each option
            options: List[str] = []
            possible = row.get('PossibleMenuOptions')
            default_opt = row.get('DefaultMenuOption')
            if pd.notna(possible):
                options += [o.strip() for o in str(possible).split(',') if o.strip()]
            if pd.notna(default_opt):
                def_opt = str(default_opt).strip()
                # include the default only if it isn't already in options
                if def_opt and def_opt not in options:
                    options.append(def_opt)
            for opt in options:
                # skip wildcard default indicator
                if opt in ('*', '✱'):
                    continue
                dest_key = f"{key_str},{opt}"
                if dest_key.strip().lower() not in keys_set:
                    warnings.append(
                        f"{key_str}: menu option '{opt}' has no matching target '{dest_key}'."
                    )
        elif act_upper in ('JUMP', 'CALLBACK', 'EMERGENCY'):
            dest = row.get('JumpToLocation')
            if pd.notna(dest):
                dest_str = str(dest).strip()
                if dest_str.lower() not in keys_set:
                    warnings.append(
                        f"{key_str}: {act_upper.lower()} target '{dest_str}' does not exist."
                    )
        elif act_upper == 'TRANSFER':
            dest = row.get('TransferTo')
            if pd.notna(dest):
                tstr = str(dest).strip()
                prefix_part = tstr.split(',', 1)[0]
                if ':' in prefix_part:
                    dest_type, dest_value = prefix_part.split(':', 1)
                    if dest_type.upper() == 'FLOW':
                        dest_value = dest_value.strip()
                        # Skip missing-flow errors for "customflow" references.  These flows
                        # are stored in separate tables and should be considered valid.
                        if dest_value.lower() not in keys_set and dest_value.lower() != 'customflow':
                            warnings.append(
                                f"{key_str}: transfer to Flow '{dest_value}' does not exist."
                            )
    return warnings


def detect_roots(df: pd.DataFrame) -> List[str]:
    """Determine root keys for the flow selector.

    A root is defined as a ``key`` value that does not end with one of
    the typical status suffixes used for ``Schedule`` branches: ``Open``,
    ``Closed`` or ``Holiday``.  This heuristic captures the initial
    entry points in most call flows.

    Args:
        df: DataFrame with a ``key`` column.

    Returns:
        Sorted list of candidate root keys.
    """
    suffixes = {'Open', 'Closed', 'Holiday'}
    roots: List[str] = []
    for k in df['key'].dropna().unique():
        if isinstance(k, float) and pd.isna(k):
            continue
        kstr = str(k)
        # Extract the last underscore-delimited segment
        if '_' in kstr:
            suffix = kstr.rsplit('_', 1)[-1]
        else:
            suffix = ''
        if suffix not in suffixes:
            roots.append(kstr)
    roots_sorted = sorted(roots)
    return roots_sorted


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="TableFlow Visualiser", layout="wide")
    st.title("TableFlow Visualiser")
    # Load default CSV if present
    default_path = Path(__file__).parent / 'Voice_StandardFlow09102025.csv'
    if default_path.exists():
        default_df = pd.read_csv(default_path, sep=None, engine='python')
    else:
        default_df = pd.DataFrame()
    # File upload
    uploaded = st.file_uploader("Upload a TableFlow CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded, sep=None, engine='python')
    else:
        df = default_df.copy()
    if df.empty:
        st.info("No CSV loaded. Please upload a file.")
        return
    # Basic validation of expected columns
    expected_cols = [
        'key', 'Action', 'TransferTo', 'JumpToLocation',
        'PossibleMenuOptions', 'DefaultMenuOption', 'ScheduleGroup'
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return
    # Editor
    st.subheader("Edit CSV")
    edited_df = st.data_editor(df, width='stretch', num_rows="dynamic")

    # Perform validation on the edited data.  Collect any warnings
    # produced by validate_flows and surface them to the user.  Each
    # warning is rendered as a separate bullet point for readability.
    warnings = validate_flows(edited_df)
    if warnings:
        warning_lines = "\n".join(f"- {html.escape(w)}" for w in warnings)
        st.warning(
            f"**Data validation warnings:**\n\n{warning_lines}",
            icon="⚠️"
        )
    else:
        st.success("No validation issues detected.")

    # Detect roots
    roots = detect_roots(edited_df)
    if not roots:
        st.info("No root flows detected. Check your CSV.")
        return
    selected_root = st.selectbox("Select a flow to visualise", roots)
    if selected_root:
        st.subheader(f"Flow: {selected_root}")
        mermaid = build_mermaid(edited_df, selected_root)
        # Render mermaid via HTML component
        # Wrap the Mermaid diagram in a container and enable pan/zoom via Panzoom.
        # We embed the panzoom library from a CDN and attach it to the container.
        mermaid_html = f"""
        <div id="mermaid-container" style="width:100%; height:600px; overflow:hidden;">
          <div class="mermaid">
            {html.escape(mermaid)}
          </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/panzoom@9.4.0/dist/panzoom.min.js"></script>
        <script>
          // Initialise mermaid when the script loads
          mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
          // After the DOM and Mermaid are ready, attach panzoom to the container
          document.addEventListener('DOMContentLoaded', function () {{
            var container = document.getElementById('mermaid-container');
            if (container) {{
              // Apply panzoom with sensible zoom limits
              panzoom(container, {{ maxZoom: 5, minZoom: 0.2 }});
            }}
          }});
        </script>
        """
        # Render the HTML component.  Enable scrolling so that oversized diagrams
        # can be navigated with the mousewheel plus pan/zoom functionality.
        st.components.v1.html(mermaid_html, height=600, scrolling=True)
    # Download edited CSV
    csv_bytes = edited_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download edited CSV", data=csv_bytes, file_name="edited_tableflow.csv", mime="text/csv")


if __name__ == '__main__':
    main()