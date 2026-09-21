"""Generate medgemma VGS answers on VQA-RAD.

Run: python vgs_medgemma_vqarad.py --device cuda:0 --limit 3 \
         --output-dir outputs/medgemma-smoke

Authentication uses your local Hugging Face login or HF_TOKEN environment
variable. HF_Token in the setup guide is a placeholder, never a credential.
"""

from vgs_decoding import main as run_cli


def main() -> None:
    run_cli("medgemma")


if __name__ == "__main__":
    main()
