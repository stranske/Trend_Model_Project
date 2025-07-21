import importlib
import sys

def test_gui_module(monkeypatch):
    class Dummy:
        def __init__(self):
            self.titles = []
            self.texts = []
        def title(self, txt):
            self.titles.append(txt)
        def write(self, txt):
            self.texts.append(txt)
    dummy = Dummy()
    monkeypatch.setitem(sys.modules, "streamlit", dummy)
    importlib.import_module("trend_analysis.gui.app")
    assert dummy.titles and dummy.texts
