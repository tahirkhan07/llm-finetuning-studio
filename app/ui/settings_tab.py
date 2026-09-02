"""
Settings Tab — HuggingFace authentication and app-level configuration.

Provides in-app token login so users can access gated models (Llama 3, Gemma, etc.)
without needing to run `huggingface-cli login` from the terminal.
"""

import os
import gradio as gr


def _check_auth_status():
    """
    Check current HuggingFace authentication status.
    Returns (is_authenticated, status_message).
    """
    try:
        from huggingface_hub import whoami
        info = whoami()
        username = info.get("name", "unknown")
        # Determine token type from auth info
        auth = info.get("auth", {})
        token_type = auth.get("accessToken", {}).get("role", "unknown")
        return True, (
            f"✅ **Authenticated as [`{username}`](https://huggingface.co/{username})**\n\n"
            f"Token type: `{token_type}` · "
            f"You can access gated models that `{username}` has been granted access to."
        )
    except Exception:
        return False, (
            "🔒 **Not authenticated.**\n\n"
            "Paste your HuggingFace access token below to access gated models "
            "(Llama 3, Gemma, etc.).\n\n"
            "Don't have a token? "
            "[Create one here →](https://huggingface.co/settings/tokens)"
        )


def build():
    with gr.Tab("⚙️ Settings"):
        gr.Markdown("### HuggingFace Authentication")
        gr.Markdown(
            "Authenticate with HuggingFace to download gated models like "
            "**Meta-Llama-3**, **Gemma**, and others that require license acceptance."
        )

        # ── Auth status display ─────────────────────────────────────────
        # Check if already authenticated (e.g. via huggingface-cli login or env var)
        _initial_auth, _initial_msg = _check_auth_status()

        # Auto-login from HF_TOKEN env var if not already authenticated
        if not _initial_auth:
            env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if env_token:
                try:
                    from huggingface_hub import login, whoami
                    # Validate before logging in
                    whoami(token=env_token)
                    login(token=env_token, add_to_git_credential=False)
                    _initial_auth, _initial_msg = _check_auth_status()
                    _initial_msg += "\n\n*🔑 Auto-logged in from `HF_TOKEN` environment variable.*"
                except Exception:
                    _initial_msg += (
                        "\n\n> ⚠️ Found `HF_TOKEN` environment variable but it appears to be "
                        "invalid or expired."
                    )

        auth_status = gr.Markdown(value=_initial_msg)

        # ── Token input ─────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                token_input = gr.Textbox(
                    label="HuggingFace Access Token",
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    type="password",
                    info="Your token is sent directly to the HuggingFace API and cached locally. It is never logged or stored by this app.",
                    interactive=True,
                )
            with gr.Column(scale=1):
                login_btn = gr.Button("🔑 Login", variant="primary", size="lg")
                logout_btn = gr.Button("🚪 Logout", variant="secondary")
                check_btn = gr.Button("🔄 Check Status")

        def do_login(token):
            """Validate and login with the provided token."""
            if not token or not token.strip():
                return (
                    "❌ **Error:** Please paste a token. "
                    "[Create one here →](https://huggingface.co/settings/tokens)"
                )

            token = token.strip()

            # Validate token format (HF tokens start with hf_)
            if not token.startswith("hf_"):
                return (
                    "❌ **Error:** Invalid token format. "
                    "HuggingFace tokens start with `hf_`. "
                    "[Create one here →](https://huggingface.co/settings/tokens)"
                )

            # Validate token against the API before caching
            try:
                from huggingface_hub import whoami
                whoami(token=token)
            except Exception:
                return (
                    "❌ **Invalid or expired token.** "
                    "Please check that your token is correct and hasn't been revoked.\n\n"
                    "[Manage your tokens →](https://huggingface.co/settings/tokens)"
                )

            # Token is valid — perform login
            try:
                from huggingface_hub import login
                login(token=token, add_to_git_credential=False)
            except Exception as e:
                return f"❌ **Login failed:** {e}"

            _, status_msg = _check_auth_status()
            return status_msg

        def do_logout():
            """Logout and clear cached token."""
            try:
                from huggingface_hub import logout
                logout()
            except Exception:
                pass  # logout() may raise if no token was cached — that's fine
            return (
                "🔒 **Logged out.**\n\n"
                "You will no longer be able to access gated models until you log in again.\n\n"
                "[Create a new token →](https://huggingface.co/settings/tokens)"
            )

        def do_check():
            """Re-check current authentication status."""
            _, msg = _check_auth_status()
            return msg

        login_btn.click(fn=do_login, inputs=[token_input], outputs=[auth_status])
        logout_btn.click(fn=do_logout, inputs=[], outputs=[auth_status])
        check_btn.click(fn=do_check, inputs=[], outputs=[auth_status])
