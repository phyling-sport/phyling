#!/usr/bin/env python3
"""
Minimal OAuth + realtime example for Phyling.

Installation et exécution :
> pip install -r requirements.txt
> python minimal_oauth.py --api-url http://localhost:5001 --port 2000 --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET --device-number 10300042

Then open http://localhost:2000/ in your browser.
"""  # noqa: E501
import argparse
import secrets
import sys
from typing import Any
from typing import Dict
from urllib.parse import urlencode

import requests
from flask import Flask
from flask import redirect
from flask import render_template_string
from flask import request
from flask import session
from flask import url_for

LOGIN_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
  <head>
    <meta charset="utf-8">
    <title>Connexion Phyling</title>
    <style>
      body { font-family: sans-serif; margin: 40px; }
      main { max-width: 420px; }
      button { padding: 10px 18px; font-size: 16px; cursor: pointer; }
    </style>
  </head>
  <body>
    <main>
      <h1>Démo OAuth Phyling</h1>
      <p>Ce mini serveur expose uniquement <code>/</code> : on y affiche ce bouton, puis on revient ici après OAuth.</p>
      <button onclick="window.location.href='{{ authorize_url }}'">Se connecter avec Phyling</button>
      <p style="margin-top:2rem;font-size:13px;color:#666;">Redirigé vers&nbsp;: <code>{{ redirect_uri }}</code></p>
    </main>
  </body>
</html>
"""

ERROR_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
  <head>
    <meta charset="utf-8">
    <title>Erreur OAuth</title>
  </head>
  <body style="font-family: sans-serif; margin: 40px;">
    <h1>Impossible de poursuivre</h1>
    <pre>{{ message }}</pre>
    <a href="/">Revenir à l'écran de connexion</a>
  </body>
</html>
"""

MINIMAL_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
  <head>
    <meta charset="utf-8">
    <title>Phyling – OAuth + temps réel</title>
    <script src="https://cdn.socket.io/4.8.1/socket.io.min.js"></script>
  </head>
  <body>
    <main>
      <h1>Phyling – temps réel (OAuth)</h1>
      <div style="margin-bottom: 12px; display: flex; gap: 12px; align-items: center;">
        <span style="color: #0a0;">✅ Connecté</span>
        <button id="logout-button" style="padding: 8px 14px; cursor: pointer;">Se déconnecter</button>
      </div>
      <pre id="log" style="height: 80vh; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;"></pre>
    </main>
    <script>
      async function main() {
        const $ = (sel) => document.querySelector(sel);
        const logBox = $('#log');
        const logoutBtn = $('#logout-button');

        const apiBase = {{ api_url|tojson }};
        const accessToken = {{ access_token|tojson }};
        const clientId = {{ realtime_client_id|tojson }};
        const deviceNumber = {{ device_number|tojson }};
        const logoutUrl = {{ logout_url|tojson }};

        const authorization = `Bearer ${accessToken}`;
        const socketBase = apiBase.replace(/^http/, 'ws');
        const socket = io(socketBase, { transports: ['websocket'] });

        logoutBtn?.addEventListener('click', async () => {
          try {
            await fetch(logoutUrl, { method: 'POST', credentials: 'same-origin' });
          } finally {
            window.location.href = '/';
          }
        });

        const log = (msg, payload) => {
          logBox.textContent += `${msg}\\n${JSON.stringify(payload, null, 2)}\\n`;
          logBox.scrollTop = logBox.scrollHeight;
        };

        socket.on('connect', () => {
          log('✅ Connecté au socket', { socketId: socket.id });
          socket.emit('subscribe', { authorization, room: `app/client/${clientId}/device/list_connected` });
          socket.emit('subscribe', { authorization, room: `app/device/${deviceNumber}/ind/json/all` });
          socket.emit('subscribe', { authorization, room: `app/device/${deviceNumber}/board/status` });
          socket.emit('subscribe', { authorization, room: `app/device/${deviceNumber}/data/json/all` });
        });

        socket.onAny((event, ...args) => log(`⟵ event: ${event}`, args[0]));
        socket.on('connect_error', (err) => log('❌ Socket error', err.message || err));
      }
      main();
    </script>
  </body>
</html>
"""


def create_app(config: Dict[str, Any]) -> Flask:
    app = Flask(__name__)
    app.secret_key = "secretkeyforflasksession"
    app.config["DEMO"] = config
    app.config["OAUTH_STATES"] = set()

    @app.route("/", methods=["GET"])
    def index():
        cfg = app.config["DEMO"]
        if "oauth_session" in session:
            auth = session["oauth_session"]
            return render_template_string(
                MINIMAL_TEMPLATE,
                api_url=cfg["api_url"],
                access_token=auth["access_token"],
                realtime_client_id=auth["realtime_client_id"],
                device_number=cfg["device_number"],
                logout_url=url_for("logout"),
            )

        pending_states = app.config["OAUTH_STATES"]
        state = secrets.token_urlsafe(16)
        pending_states.add(state)

        authorize_query = urlencode(
            {
                "client_id": cfg["oauth_client_id"],
                "redirect_uri": cfg["redirect_uri"],
                "state": state,
            }
        )
        authorize_url = f"{cfg['api_url']}/oauth/authorize?{authorize_query}"

        return render_template_string(
            LOGIN_TEMPLATE,
            authorize_url=authorize_url,
            redirect_uri=cfg["redirect_uri"],
        )

    @app.route("/callback", methods=["GET"])
    def oauth_callback():
        cfg = app.config["DEMO"]
        pending_states = app.config["OAUTH_STATES"]
        code = request.args.get("code")
        incoming_state = request.args.get("state")

        if not code:
            return (
                render_template_string(
                    ERROR_TEMPLATE, message="Missing authorization code"
                ),
                400,
            )
        if not incoming_state or incoming_state not in pending_states:
            return (
                render_template_string(
                    ERROR_TEMPLATE, message="Invalid or missing state"
                ),
                400,
            )
        pending_states.discard(incoming_state)

        token_url = f"{cfg['api_url']}/oauth/token"
        try:
            token_resp = requests.post(
                token_url,
                json={
                    "client_id": cfg["oauth_client_id"],
                    "client_secret": cfg["client_secret"],
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": cfg["redirect_uri"],
                },
                timeout=10,
            )
            token_resp.raise_for_status()
            payload = token_resp.json()
        except requests.RequestException as exc:
            return (
                render_template_string(
                    ERROR_TEMPLATE, message=f"Token request failed: {exc}"
                ),
                502,
            )
        except ValueError:
            return (
                render_template_string(
                    ERROR_TEMPLATE, message="Token response is not JSON"
                ),
                502,
            )

        access_token = payload.get("access_token")
        if not access_token:
            return (
                render_template_string(
                    ERROR_TEMPLATE,
                    message=f"Missing access_token in response: {payload}",
                ),
                502,
            )

        realtime_client_id = payload.get("user", {}).get("client_id") or cfg.get(
            "fallback_client_id"
        )
        if realtime_client_id is None:
            return (
                render_template_string(
                    ERROR_TEMPLATE,
                    message="Unable to resolve realtime client_id (token.user.client_id missing)",
                ),
                500,
            )

        session["oauth_session"] = {
            "access_token": access_token,
            "realtime_client_id": realtime_client_id,
        }

        return redirect(url_for("index"))

    @app.route("/logout", methods=["POST"])
    def logout():
        session.pop("oauth_session", None)
        return ("", 204)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mini démonstration OAuth + Socket.IO Phyling"
    )
    parser.add_argument(
        "--api-url",
        required=True,
        help="URL de base de l'API Phyling (ex: http://localhost:5001)",
    )
    parser.add_argument(
        "--client-id", required=True, help="Client ID OAuth fourni par Phyling"
    )
    parser.add_argument(
        "--client-secret", required=True, help="Client secret OAuth fourni par Phyling"
    )
    parser.add_argument(
        "--port", type=int, default=2000, help="Port local d'écoute (callback sur /)"
    )
    parser.add_argument(
        "--device-number",
        type=int,
        default=10300042,
        dest="device_number",
        help="Device number à écouter après connexion",
    )
    parser.add_argument(
        "--fallback-client-id",
        type=int,
        default=None,
        dest="fallback_client_id",
        help="Client ID temps réel à utiliser si le token ne contient pas user.client_id",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_url = args.api_url.rstrip("/")
    redirect_uri = f"http://localhost:{args.port}/callback"

    config = {
        "api_url": api_url,
        "oauth_client_id": args.client_id,
        "client_secret": args.client_secret,
        "redirect_uri": redirect_uri,
        "device_number": args.device_number,
        "fallback_client_id": args.fallback_client_id,
    }
    app = create_app(config)
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False)
    except OSError as exc:
        print(f"Impossible de démarrer le serveur: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
