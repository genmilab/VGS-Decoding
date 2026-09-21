"""Generate llava-med VGS answers on VQA-RAD.

Run: python vgs_llavamed_vqarad.py --device cuda:0 --limit 3 \
         --output-dir outputs/llava-med-smoke

Authentication uses your local Hugging Face login or HF_TOKEN environment
variable. HF_Token in the setup guide is a placeholder, never a credential.
"""

from vgs_decoding import main as run_cli


def main() -> None:
    run_cli("llava-med")


if __name__ == "__main__":
    main()
