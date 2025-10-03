# Binary Asset Substitutes

The original Codex bootstrap attached DejaVu font binaries so PDF exports could render Unicode bullet points. Because this repository avoids committing binary blobs, the assets are stored here as base64 text along with their license.

## Files
- `DejaVuSans.base64` – base64 encoding of `DejaVuSans.ttf`
- `DejaVuSans-Bold.base64` – base64 encoding of `DejaVuSans-Bold.ttf`
- `DejaVu-LICENSE.md` – SIL Open Font License that applies to the bundled fonts

## Restoring the Fonts
Use the helper snippet below to regenerate the original `.ttf` files if you need to embed them in a build or Docker image:

```bash
python - <<'PY'
from pathlib import Path
import base64

root = Path('agents/codex-issue-1667-18225483679/binary-assets')
for name in ["DejaVuSans", "DejaVuSans-Bold"]:
    encoded = (root / f"{name}.base64").read_text()
    data = base64.b64decode(encoded)
    (root / f"{name}.ttf").write_bytes(data)
PY
```

After extraction, install the fonts into your packaging pipeline or reference them via `fpdf.add_font` with `uni=True`.
