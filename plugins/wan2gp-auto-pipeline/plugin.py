import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import os
import shutil
from pathlib import Path
from shared.api import init
import cv2

PlugIn_Name = "Auto Pipeline"
PlugIn_Id = "AutoPipeline"

class AutoPipelinePlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.output_dir = os.path.join(os.getcwd(), "outputs", "auto_pipeline")
        os.makedirs(self.output_dir, exist_ok=True)

    def setup_ui(self):
        self.add_tab(
            tab_id=PlugIn_Id,
            label=PlugIn_Name,
            component_constructor=self.create_ui,
        )

    def on_tab_select(self, state: dict) -> None:
        pass

    def on_tab_deselect(self, state: dict) -> None:
        pass

    def run_pipeline(self, prompt_text: str, custom_image_path: str = "", flux_loras: str = "", ltx_loras: str = ""):
        """
        Headless friendly pipeline.
        If prompt_text is a path ending with .txt, it reads the file.
        Otherwise it splits by \n\n.
        LoRA strings can be multiple separated by comma: "lora1.safetensors, lora2.safetensors"
        """
        if os.path.isfile(prompt_text) and prompt_text.lower().endswith(".txt"):
            with open(prompt_text, "r", encoding="utf-8") as f:
                prompts = [p.strip() for p in f.read().split("\n\n") if p.strip()]
        else:
            prompts = [p.strip() for p in prompt_text.split("\n\n") if p.strip()]

        if not prompts:
            return "No prompts provided to process."

        session = init(root=Path(os.getcwd()))
        
        results_log = []
        for i, prompt in enumerate(prompts):
            seq_num = f"{i+1:04d}"
            results_log.append(f"Processing sequence {seq_num}...")
            
            # Parsing LoRAs
            f_loras = [l.strip() for l in flux_loras.split(",") if l.strip()]
            l_loras = [l.strip() for l in ltx_loras.split(",") if l.strip()]

            # Step 1: Image Generation (Flux 2 Klein 9B)
            img_path = None
            if custom_image_path and os.path.isfile(custom_image_path) and len(prompts) == 1:
                # Use provided image if it's a 1-to-1 prompt
                img_path = custom_image_path
                results_log.append(f"{seq_num}: Using provided image: {custom_image_path}")
            else:
                flux_settings = {
                    "model_type": "flux2_klein_9b",
                    "prompt": prompt,
                    "resolution": "1024x1024",
                    "num_inference_steps": 4,
                }
                if f_loras:
                    flux_settings["loras"] = f_loras

                # Submit Flux generation
                img_job = session.submit_task(flux_settings)
                img_res = img_job.result()
                
                if img_res.success and img_res.generated_files:
                    raw_img_path = img_res.generated_files[0]
                    raw_ext = os.path.splitext(raw_img_path)[1].lower()
                    img_path = os.path.join(self.output_dir, f"{seq_num}_flux_image.png")

                    if raw_ext in (".mp4", ".avi", ".mov", ".mkv"):
                        # WanGP returned a video container – extract the first frame as PNG
                        results_log.append(f"{seq_num}: API returned video file ({raw_ext}), extracting first frame...")
                        cap = cv2.VideoCapture(raw_img_path)
                        ret, frame = cap.read()
                        cap.release()
                        if ret:
                            cv2.imwrite(img_path, frame)
                            results_log.append(f"{seq_num}: First frame extracted -> {img_path}")
                        else:
                            results_log.append(f"{seq_num}: Failed to extract frame from returned video. Skipping.")
                            continue
                    elif raw_ext in (".png", ".jpg", ".jpeg", ".webp"):
                        shutil.copy2(raw_img_path, img_path)
                        results_log.append(f"{seq_num}: Image generated -> {img_path}")
                    else:
                        # Unknown extension – copy as-is and warn
                        img_path = os.path.join(self.output_dir, f"{seq_num}_flux_image{raw_ext}")
                        shutil.copy2(raw_img_path, img_path)
                        results_log.append(f"{seq_num}: Image generated (unknown ext '{raw_ext}') -> {img_path}")
                else:
                    results_log.append(f"{seq_num}: Failed to generate image.")
                    if img_res.errors:
                        results_log.append(f"Error: {img_res.errors[0].message}")
                    continue

            # Step 2: Video Generation (LTX 2.3 Distilled)
            # 10s video = 241 frames at 24 fps
            ltx_settings = {
                "model_type": "ltx2_22B_distilled",
                "prompt": prompt,
                "resolution": "1280x720",
                "num_inference_steps": 8,
                "video_length": 241,  
                "force_fps": 24,
                "image_refs": [img_path],
                "video_prompt_type": "", 
                "guide_custom_choices": "KFI", # Inject Frames (First, Last, Middle)
                "custom_frames_injection": True
            }
            if l_loras:
                ltx_settings["loras"] = l_loras

            vid_job = session.submit_task(ltx_settings)
            vid_res = vid_job.result()
            
            if vid_res.success and vid_res.generated_files:
                raw_vid_path = vid_res.generated_files[0]
                ext = os.path.splitext(raw_vid_path)[1]
                vid_path = os.path.join(self.output_dir, f"{seq_num}_ltx_video{ext}")
                shutil.copy2(raw_vid_path, vid_path)
                results_log.append(f"{seq_num}: Video generated -> {vid_path}")
            else:
                results_log.append(f"{seq_num}: Failed to generate video.")
                if vid_res.errors:
                    results_log.append(f"Error: {vid_res.errors[0].message}")

        return "\n".join(results_log)


    def create_ui(self):
        with gr.Column():
            gr.Markdown("### Auto Pipeline (Flux 2 Klein 9B Image -> LTX 2.3 Distilled Video)")
            gr.Markdown(
                "1. **Input**: Enter multiple prompts separated by an empty line, OR provide an absolute path to a `.txt` file.\n"
                "2. **Process**: \n"
                "   - Automatically generates an image for each prompt using `Flux 2 Klein 9B`.\n"
                "   - Takes the generated image as a starting frame (first/inject frames). \n"
                "   - Automatically generates a 10s video using `LTX 2.3 Distilled`.\n"
                "   - Output files are sequenced and saved in `outputs/auto_pipeline/`."
            )
            
            prompts_input = gr.Textbox(
                label="Prompts (separated by empty line) or Path to .txt file", 
                lines=10, 
                placeholder="Prompt 1...\n\nPrompt 2...\n\nPrompt 3..."
            )
            
            with gr.Row():
                flux_loras_input = gr.Textbox(
                    label="Optional: Flux LoRAs (comma separated paths or names)",
                    lines=1,
                    placeholder="my_flux_lora.safetensors"
                )
                ltx_loras_input = gr.Textbox(
                    label="Optional: LTX LoRAs (comma separated paths or names)",
                    lines=1,
                    placeholder="my_ltx_lora.safetensors"
                )

            custom_image_input = gr.Textbox(
                label="Optional: Absolute path to custom start image (used if only 1 prompt provided)",
                lines=1,
            )

            run_btn = gr.Button("Run Auto Pipeline", variant="primary")
            
            status_output = gr.Textbox(label="Pipeline Output Status", lines=12, interactive=False)
            
        self.on_tab_outputs = []

        run_btn.click(
            fn=self.run_pipeline,
            inputs=[prompts_input, custom_image_input, flux_loras_input, ltx_loras_input],
            outputs=[status_output],
            api_name="auto_pipeline"
        )
