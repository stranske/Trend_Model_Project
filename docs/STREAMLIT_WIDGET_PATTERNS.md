# Streamlit Widget State Patterns

This document captures hard-won lessons about Streamlit widget state management discovered during development of the Data page.

## The Core Problem

Streamlit widget keys can get deleted between reruns if the widget doesn't render on a given run. This causes user selections to be lost.

## Pattern That Works (for selectbox, radio, etc.)

```python
# 1. Calculate index from current canonical state
current_bench = st.session_state.get("selected_benchmark")
bench_idx = options.index(current_bench) if current_bench in options else 0

# 2. Define on_change callback that updates canonical state
def on_bench_change():
    val = st.session_state["_widget_bench"]
    st.session_state["selected_benchmark"] = val

# 3. Render widget with index= and on_change=
st.selectbox(..., index=bench_idx, key="_widget_bench", on_change=on_bench_change)
```

### Why this works:
- `index=` sets the visual state based on canonical state
- `on_change` fires BEFORE rerun, capturing user input into canonical state
- On next render, `index=` recalculates from the (now updated) canonical state
- The canonical state (`selected_benchmark`) persists even if widget key is deleted

## Patterns That DON'T Work

### 1. Setting widget key before render
```python
# BAD - gets overwritten or deleted
st.session_state["_widget_key"] = current_value
st.selectbox(..., key="_widget_key")
```

### 2. Using `default=` with `key=`
```python
# BAD - default= overrides user changes on every render
st.multiselect(..., default=current_list, key="_widget_key")
```

### 3. Relying on widget key to persist
```python
# BAD - keys get deleted when widget doesn't render
if "_widget_key" not in st.session_state:
    st.session_state["_widget_key"] = initial_value
st.selectbox(..., key="_widget_key")  # key may not exist on next render!
```

### 4. Using return value without on_change
```python
# BAD - value reflects index= not user input
choice = st.selectbox(..., index=idx)
st.session_state["selection"] = choice  # This captures idx, not user choice
```

## For data_editor (checkbox tables)

The `on_change` callback receives a **dict of deltas** in session state, NOT the full DataFrame.

### Delta format:
```python
{
    "edited_rows": {row_idx: {"column_name": new_value}},
    "added_rows": [],
    "deleted_rows": []
}
```

### Correct pattern:
```python
# Canonical state
if "selected_fund_columns" not in st.session_state:
    st.session_state["selected_fund_columns"] = set(available_funds)

current_selection = st.session_state.get("selected_fund_columns", set())

# Build DataFrame from canonical state
fund_df = pd.DataFrame({
    "Include": [fund in current_selection for fund in available_funds],
    "Fund Name": available_funds,
})

# on_change processes deltas
def on_editor_change():
    editor_state = st.session_state.get("_fund_editor", {})
    edited_rows = editor_state.get("edited_rows", {})
    
    new_selection = set(st.session_state.get("selected_fund_columns", []))
    
    for row_idx, changes in edited_rows.items():
        if "Include" in changes:
            fund_name = available_funds[row_idx]
            if changes["Include"]:
                new_selection.add(fund_name)
            else:
                new_selection.discard(fund_name)
    
    st.session_state["selected_fund_columns"] = new_selection

st.data_editor(fund_df, key="_fund_editor", on_change=on_editor_change, ...)
```

### Bulk buttons:
```python
if st.button("Select All"):
    st.session_state["selected_fund_columns"] = set(available_funds)
    st.rerun()  # Rebuilds DataFrame from updated canonical state
```

## Key Principles

1. **Always have canonical state** - A session state key that persists independently of widgets
2. **Use `on_change` callbacks** - They fire BEFORE rerun, capturing user intent
3. **Rebuild widget state from canonical state** - Use `index=` or build DataFrames fresh each render
4. **Never rely on widget keys persisting** - They can be deleted if widget doesn't render
5. **For bulk operations** - Update canonical state directly, then `st.rerun()`
