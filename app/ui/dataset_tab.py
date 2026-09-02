import gradio as gr
from app.state import state
from core.datasets.loader import load_dataset
from core.datasets.mapper import map_to_canonical, _is_preformatted
from core.datasets.validator import validate_dataset

def build():
    with gr.Tab("2. Dataset"):
        gr.Markdown("### Ingest & Map Dataset")
        
        with gr.Row():
            with gr.Column():
                ds_source = gr.Textbox(
                    label="Dataset Source (HF ID or local path)",
                    placeholder="tatsu-lab/alpaca",
                    info="HuggingFace dataset ID, or path to a local CSV/JSON/JSONL file."
                )
                load_ds_btn = gr.Button("Load Dataset")
                
                instr_col = gr.Dropdown(label="Instruction Column", choices=[])
                input_col = gr.Dropdown(label="Input Column (Optional)", choices=[])
                output_col = gr.Dropdown(label="Output/Response Column", choices=[])
                sys_prompt = gr.Textbox(label="System Prompt", value="You are a helpful AI assistant.")
                
                map_btn = gr.Button("Map & Validate", variant="primary", interactive=False)
                
            with gr.Column():
                ds_status = gr.Markdown("*No dataset loaded.*")
                preview_box = gr.Markdown(visible=False)
                val_report = gr.JSON(label="Validation Report", visible=False)

        def _guess_column(cols, candidates, fallback_index=None):
            """Try to auto-guess a column from a list of candidate names."""
            for c in candidates:
                if c in cols:
                    return c
            if fallback_index is not None and len(cols) > fallback_index:
                return cols[fallback_index]
            return None

        def handle_load(source):
            try:
                ds = load_dataset(source)
                state.raw_dataset = ds
                cols = ds.column_names if isinstance(ds.column_names, list) else list(ds.features.keys())
                
                # Check if dataset is already pre-formatted
                if _is_preformatted(ds):
                    return (
                        f"✅ **Loaded dataset** with {len(ds)} rows.\n\n"
                        f"🎉 **Auto-detected pre-formatted chat dataset!** "
                        f"This dataset already has a `messages`/`conversations` column. "
                        f"Column mapping will be skipped automatically — just click **Map & Validate**.",
                        gr.update(choices=cols, value=cols[0] if cols else None),
                        gr.update(choices=["None"] + cols, value="None"),
                        gr.update(choices=cols, value=cols[-1] if cols else None),
                        gr.update(interactive=True),
                    )
                
                # Auto-guess columns based on common naming patterns
                instr_guess = _guess_column(cols, ["instruction", "question", "prompt", "input", "text", "quote", "sentence"], 0)
                in_guess = _guess_column(cols, ["input", "context"])
                # Avoid guessing the same column for both instruction and input
                if in_guess == instr_guess:
                    in_guess = None
                out_guess = _guess_column(cols, ["output", "response", "answer", "completion", "tags", "label", "target"], -1)
                
                # Build a preview of the first 3 rows
                preview_lines = [f"**Columns:** `{'`, `'.join(cols)}`\n"]
                for i in range(min(3, len(ds))):
                    row = ds[i]
                    preview_lines.append(f"**Row {i}:** " + " | ".join(
                        f"`{k}`={repr(str(v)[:80])}" for k, v in row.items()
                    ))
                preview_text = "\n".join(preview_lines)
                
                return (
                    f"✅ **Loaded dataset** with {len(ds)} rows.",
                    gr.update(choices=cols, value=instr_guess),
                    gr.update(choices=["None"] + cols, value=in_guess or "None"),
                    gr.update(choices=cols, value=out_guess),
                    gr.update(interactive=True),
                )
            except Exception as e:
                return (
                    f"❌ **Error:** {str(e)}",
                    gr.update(), gr.update(), gr.update(),
                    gr.update(interactive=False),
                )

        def handle_map(instr, inp, out, sys_p):
            if state.raw_dataset is None:
                return "No dataset loaded.", gr.update()
            
            try:
                inp_col = None if inp in ("None", "", None) else inp
                canonical = map_to_canonical(state.raw_dataset, instr, inp_col, out, sys_p)
                report = validate_dataset(canonical, state.tokenizer)
                
                if report.invalid_samples > 0:
                    # Auto-filter: keep only rows where every message has non-empty content
                    def is_valid_row(row):
                        for msg in row["messages"]:
                            content = msg.get("content", "")
                            if not isinstance(content, str) or content.strip() == "":
                                return False
                        return True

                    clean = canonical.filter(is_valid_row)
                    state.canonical_dataset = clean
                    
                    return (
                        f"✅ **Dataset mapped & cleaned!** "
                        f"Removed {report.invalid_samples} rows with empty content. "
                        f"**{len(clean)}** samples ready for training.",
                        gr.update(value=report.model_dump(), visible=True),
                    )
                else:
                    state.canonical_dataset = canonical
                    return (
                        f"✅ **Dataset mapping successful & validated!** "
                        f"**{len(canonical)}** samples ready for training.",
                        gr.update(value=report.model_dump(), visible=True),
                    )
            except Exception as e:
                return f"❌ **Error during mapping:** {str(e)}", gr.update(visible=False)

        load_ds_btn.click(
            fn=handle_load,
            inputs=[ds_source],
            outputs=[ds_status, instr_col, input_col, output_col, map_btn]
        )
        
        map_btn.click(
            fn=handle_map,
            inputs=[instr_col, input_col, output_col, sys_prompt],
            outputs=[ds_status, val_report]
        )
