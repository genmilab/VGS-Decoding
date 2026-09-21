# Project website

The static project page is `docs/index.html`. It uses plain HTML, CSS, and
JavaScript: no build step, analytics, remote fonts, or JavaScript dependencies.
Its original design follows the academic project-page structure of the CCD
reference supplied by the project owner; no CCD text, figures, or code are copied.

The website presents the abstract, author affiliations and equal-first-author
credit, VGS method, three main benchmark tables, all supplied figure assets,
17 expandable appendix tables, and six qualitative case studies. The code section links to
`vgs_llavamed_vqarad.py`, `vgs_medgemma_vqarad.py`, and `vgs_decoding.py`.
The interactive slider illustrates the probability formula with synthetic
probabilities. Code contributors are `govindakolli` and `adinathdukre`.

The separate Demo section links to `hf_demo/` while Hugging Face hosting and
GPU access are pending. It is distinct from the synthetic formula slider.
Replace its destination and status only after a Space has been deployed and
tested; do not invent a live Hugging Face URL.

## Preview

From the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8000`. If running on a remote server, forward port 8000
over SSH first. Do not expose a directory containing credentials or datasets.

## Publish on GitHub Pages

For repository creation, separate Govinda/Adinath access, SSH authentication,
and the initial push, follow [the two-contributor setup guide](GITHUB_SETUP.md).
The organization handle is `genmilab`; **GenMI-Lab** is its display name.

After the public repository `genmilab/VGS-Decoding` exists:

1. Open repository **Settings → Pages**.
2. Select **Deploy from a branch**.
3. Select **main** and the **/docs** folder, then save.
4. Check the Pages deployment before announcing the site as live.

The expected URL is `https://genmilab.github.io/VGS-Decoding/`; this document
does not imply that the repository or website has already been published.

## Content and version handling

- The project author explicitly requested seven authors and equal-first-author
  markers for Govinda Kolli and Adinath Madhavrao Dukre. Affiliations for five
  authors are verified against arXiv; the author confirmed MBZUAI for both
  Yifan Lu and Ziyun Zou and requested MBZUAI only for Govinda Kolli. The
  website now lists three institutions with consistent affiliation numbers.
- Paper buttons link to arXiv:2603.20314. BibTeX preserves that preprint's
  five-author record. No conference acceptance is claimed.
- The results are identified as belonging to the supplied extended manuscript,
  which includes SLAKE and appendix material beyond the linked arXiv v1.
- All twelve supplied figure assets are retained. The two PDF assets are also
  rendered as PNG for browsers. See `CONTENT_SOURCES.md` and `NOTICE.md` for
  content versions, corrections, and third-party rights.
- The lab logo in the hero and footer is a locally stored, unchanged copy of
  the public `genmilab` organization avatar. Both placements link to the lab.

Edit text and links in `index.html`, presentation in `assets/styles.css`, and
the formula illustration, result tabs, and command tabs in `assets/site.js`.
The copy buttons and both tab sets have keyboard support. All three results
tables remain visible without JavaScript. The page respects reduced
motion preferences and retains its main content when JavaScript is disabled.
