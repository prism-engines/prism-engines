"""
PRISM Plugins
==============

Drop-in plugin directories for engines, workflows, and panels.

To create a plugin:
1. Create a Python file in the appropriate directory
2. Define a class that extends the appropriate base class
3. The plugin will be auto-discovered!

Example engine plugin:
    # plugins/engines/my_engine.py
    from core import EnginePlugin

    class MyEngine(EnginePlugin):
        name = "my_engine"
        version = "1.0.0"

        def analyze(self, data, **kwargs):
            return {"status": "ok"}
"""
