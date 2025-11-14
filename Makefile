.PHONY: lock

lock:
	uv pip compile pyproject.toml --extra app --extra dev --extra notebooks -o requirements.lock
