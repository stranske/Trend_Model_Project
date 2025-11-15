.PHONY: lock

# Requires uv: pip install uv
lock:
	@command -v uv >/dev/null 2>&1 || { echo >&2 "Error: 'uv' is not installed. Please run 'pip install uv'."; exit 1; }
	uv pip compile pyproject.toml --extra app --extra dev --extra notebooks -o requirements.lock
