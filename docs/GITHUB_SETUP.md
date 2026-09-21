# Publish under genmilab with two contributors

Repository owner: **genmilab**, the GitHub organization whose display name is
**GenMI-Lab**. Use the account handle in URLs, not the display name.

- Repository: `https://github.com/genmilab/VGS-Decoding`
- SSH remote: `git@github.com:genmilab/VGS-Decoding.git`
- Project website: `https://genmilab.github.io/VGS-Decoding/`
- Contributors: `govindakolli` and `adinathdukre`, using separate personal logins.

The initial public release and GitHub Pages deployment were verified on
21 September 2026. The setup instructions below are retained for reference;
do not repeat repository creation or the initial commit in the existing checkout.
For subsequent updates, review the intended changes, commit, and push normally.
Hugging Face hosting is separate and remains pending.

## 1. Create the repository under the lab

An organization owner, or member allowed to create repositories, should:

1. Sign in with their own GitHub account and open [New repository](https://github.com/new?owner=genmilab&name=VGS-Decoding&visibility=public).
2. Verify **Owner = genmilab**, **Repository name = VGS-Decoding**, and
   **Visibility = Public** before submitting the form.
3. For an initial push of this prepared project, leave automatic README,
   `.gitignore`, and license initialization off; those files are already here.
4. Create the repository. If it already exists and contains commits, stop and
   inspect that history; do not force-push or replace it with unrelated history.

If `genmilab` is unavailable in the Owner selector, ask an existing organization
owner to create the repository or grant the appropriate creation permission.
Do not create another organization or share an organization's credentials.
See [GitHub's repository creation guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository).

## 2. Give both personal accounts access

The repository admin/organization owner opens the repository's
**Settings → Collaborators and teams** (under Access; wording may appear as
**Manage access**) and uses **Add people** to add:

| GitHub account | Permission needed for pushing |
| -------------- | ----------------------------- |
| `govindakolli` | Write                         |
| `adinathdukre` | Write                         |

Each person must accept the invitation using the matching personal account.
Existing organization members can receive repository access directly or through
a team; nonmembers can be invited as outside collaborators if lab policy allows.
Organization membership by itself does not guarantee write access to this repo.

**Write** is sufficient for code contributions. **Maintain** or **Admin** is
needed to configure Pages; **Admin** is needed to manage repository access.
An existing organization owner can handle setup without making both contributors
organization owners. See [repository roles](https://docs.github.com/en/organizations/managing-user-access-to-your-organizations-repositories/managing-repository-roles/repository-roles-for-an-organization)
and [access management](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-teams-and-people-with-access-to-your-repository).

## 3. Authenticate separately

Each person should use their own machine or separate operating-system account,
their own checkout, and an SSH key associated with their own GitHub account.
Never share a private key, personal access token, password, or authenticated
working directory. On a shared server, do not switch identities by handing
one user's private key to the other.

If an SSH key is not already configured, follow GitHub's
[key generation guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).
Do not overwrite an existing key. Add only its **public `.pub` key** under your
personal GitHub **Settings → SSH and GPG keys → New SSH key**; never upload the
private key. Follow any organization SSO/2FA requirements.

Test from each person's shell:

```bash
ssh -T git@github.com
```

Verify the first-connection host fingerprint against GitHub's documentation.
The greeting must say `Hi govindakolli!` for Govinda and `Hi adinathdukre!` for
Adinath. GitHub's successful test normally exits with code 1 because it does not
provide shell access. A wrong greeting means the wrong identity is being used;
fix SSH selection before pushing. See [SSH connection testing](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection).

Git's `user.name` and `user.email` control **commit credit**, not authentication
or access. Use your own verified email or copy your exact GitHub-provided
`noreply` address from personal **Settings → Emails**. Do not guess its format.
See [commit email configuration](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address).

## 4. Initial upload: one person only

Below, Adinath performs the first upload after the repository is created empty,
the invitation is accepted, and SSH authenticates as `adinathdukre`.

Open the prepared `VGS-Decoding` directory, not the full research workspace.
If using the ZIP, extract it first and enter its `VGS-Decoding` directory.
For the ZIP copy only, run `git init -b main` if it has no `.git` directory.
The prepared workspace repository is already initialized on `main`.

```bash
git status
git remote -v
git config --local user.name "Adinath Madhavrao Dukre"
git config --local user.email "REPLACE_WITH_YOUR_ADINATH_GITHUB_EMAIL"

git add .gitattributes .gitignore .dockerignore .github LICENSE NOTICE.md README.md pyproject.toml docs hf_demo vgs_decoding.py vgs_llavamed_vqarad.py vgs_medgemma_vqarad.py
git diff --cached --stat
git diff --cached --check
git commit -m "Release VGS-Decoding inference code and project website"
git remote add origin git@github.com:genmilab/VGS-Decoding.git
git push -u origin main
```

Replace the email placeholder before running these commands. They are intended
for an empty remote and a local repository with no `origin`. If `origin` already
exists, inspect its URL rather than blindly adding or replacing it. If a push
is rejected, investigate the message; do not use `--force` as a workaround.
Only one contributor performs this initial upload; the other clones afterward.
Do not fabricate another person's authorship or email to create contributor
credit. Both names are already credited in the README and website.

## 5. Govinda clones; both use their own branches

Govinda runs this on his own machine/account after the initial upload:

```bash
git clone git@github.com:genmilab/VGS-Decoding.git
cd VGS-Decoding
git config --local user.name "Govinda Kolli"
git config --local user.email "REPLACE_WITH_YOUR_GOVINDA_GITHUB_EMAIL"
git switch -c govinda/update-docs
```

After making an actual change, stage its intended files, commit, and push:

```bash
git add docs/WEBSITE.md
git commit -m "Clarify website documentation"
git push -u origin govinda/update-docs
```

Open a pull request to `genmilab/VGS-Decoding:main`. Adinath follows the same
pattern with branches such as `adinath/update-docs`. Before starting later work,
use `git switch main` and `git pull --ff-only` with a clean working tree, then
create a new branch. Both use the **same lab remote**, not separate personal
repositories. Review and merge the pull request before expecting the site to
change. Normal branch protections still apply even with Write access.

## 6. Enable the project website

After `main` and `docs/index.html` exist on GitHub, a maintainer/admin opens
**Settings → Pages → Build and deployment**:

1. Source: **Deploy from a branch**.
2. Branch: **main**.
3. Folder: **/docs**.
4. Save and check the Pages deployment result in **Actions**.

The expected URL is `https://genmilab.github.io/VGS-Decoding/`, unless the lab
configures a custom domain. This is a project site; a separate
`genmilab.github.io` repository is not required. Merged changes under `docs/`
are published from `main`. See [GitHub Pages publishing sources](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

Before announcing it, confirm the live page, logo, Paper link, three code links,
and mobile tables load correctly. Keep all real Hugging Face credentials local;
GitHub access does not require sharing MedGemma credentials.
