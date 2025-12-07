Let’s do the correct stable fix now (recommended)

Ask Claude Code to apply THIS patch exactly:

if __name__ == "__main__":
-    app.run(debug=True)
+    import argparse, os
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--port", type=int, default=None)
+    parser.add_argument("--host", type=str, default="127.0.0.1")
+    args = parser.parse_args()
+
+    env_port = os.environ.get("FLASK_RUN_PORT")
+    port = args.port or (int(env_port) if env_port else 5000)
+
+    app.run(host=args.host, port=port, debug=True, use_reloader=False)


After that:

python app.py --port 5050


will work exactly as expected.
