# Deploying the Model 1 API to a Hugging Face Space

Target: a public URL the ISAs can open, with the trained weights kept private.

Three things only you can do, because they involve creating an account and
handling a token. Everything else is already built.

## 1. Put the checkpoints in a private model repo

On https://huggingface.co, **New** → **Model**. Name it something like
`armor-model1-checkpoints`. Set visibility to **Private**.

Upload the ten files, keeping the directory structure:

```
model1_v1/seed_0/s2_best.pt
model1_v1/seed_0/s2_best.json
...
model1_v1/seed_4/s2_best.pt
model1_v1/seed_4/s2_best.json
```

They are on this machine at `C:/Users/bhoom/298A_project/checkpoints/model1_v1/`,
3.2 MB in total. The web uploader handles a drag-and-drop of the `model1_v1`
folder.

## 2. Create a read token

**Settings** → **Access Tokens** → **New token**, type **Read**. Copy it.

Do not paste it into a file in this repository. The repository is public.

## 3. Create the Space and push

**New** → **Space**. SDK **Docker**, template **Blank**, hardware **CPU basic
(free)**, visibility **Public**.

Then in the Space's **Settings**:

| Where | Name | Value |
|---|---|---|
| Variables | `MODEL1_HF_REPO` | `<your-username>/armor-model1-checkpoints` |
| Secrets | `HF_TOKEN` | the read token from step 2 |

A *variable* is visible to anyone; a *secret* is not. The repo name is fine as a
variable. The token must be a secret.

Now assemble and push:

```bash
./build_space.sh ~/armor-model1-space
```

```bash
cd ~/armor-model1-space && git init -b main && git add -A && git commit -m "Deploy Model 1 v2 detection API"
```

```bash
git remote add origin https://huggingface.co/spaces/<your-username>/<space-name> && git push -u origin main
```

Git will ask for a username and password. The password is a **write** token, not
your account password.

The first build takes 10 to 15 minutes, almost all of it installing torch. Watch
the **Logs** tab. `build_space.sh` refuses to run if it finds model weights in
the Space directory, so the weights cannot be pushed by accident.

## Checking it worked

```bash
curl https://<your-username>-<space-name>.hf.space/healthz
```

Expect `{"status": "ok", "model": "model1-v2-ensemble", "version": "2.0.0"}`.

Then confirm it loaded the frozen weights rather than something else:

```bash
curl -s https://<your-username>-<space-name>.hf.space/v1/metadata
```

The five `checkpoint_sha256` values must start `01940bfb`, `671e0fb3`,
`1c29d48b`, `cb2c31a7`, `c25f2fa0`. Those are the hashes in
`armor-system/models/model1_transformer/results/test_freeze/evidence_test_freeze.json`.
If they match, the Space is serving the same model the test results were
measured on.

And end to end against real recorded telemetry:

```bash
py -3.13 -I armor-system/models/model1_v2/serving/replay_demo.py --url https://<your-username>-<space-name>.hf.space
```

That should report TP 18, FP 6, F1 0.7347, the committed validation numbers.
It sends 15 requests of about 5 MB each, so give it a few minutes.

## Things that will go wrong

**Space shows "Configuration error".** The `README.md` front matter is what
Spaces reads for `sdk` and `app_port`. It must be the very first thing in the
file, between `---` lines.

**Build succeeds, Space stays "Building" then errors.** Check the Logs tab for
`MODEL1_HF_REPO is not set`. The variable has to be set in Space settings, not
in the Dockerfile.

**`401` or `403` in the logs during start-up.** The `HF_TOKEN` secret is missing,
expired, or is a token for a different account than the one owning the private
repo.

**Service starts then exits.** `service.py` refuses to run if a checkpoint hash
does not match the frozen evidence. The log names the file. Most likely the
upload in step 1 lost the directory structure, so re-check the paths are
`model1_v1/seed_N/...` and not flattened.

**It is slow.** Free Spaces sleep after 48 hours idle and take about 30 seconds
to wake, then 7 seconds to hash the checkpoints. Open the URL a few minutes
before the demo so it is warm.

## What this is not

A free CPU Space is not production hosting. It sleeps, it has no SLA, and it
runs one request at a time. It is the right tool for a demo URL and the wrong
tool for anything that matters later. Real deployment still needs the four team
decisions in `docs/team_handoff_readiness_audit.md` section 7: provider, object
storage, credential handling for a public repository, and budget.
