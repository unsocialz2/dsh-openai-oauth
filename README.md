# dsh-openai-oauth

English | [简体中文](README.zh-CN.md)

Use GPT models from your ChatGPT account as the main model in DeepSeek Harness. Harness keeps its own agent loop and runs its own tools. This package connects its LLM interface to the official local Codex app-server, which handles ChatGPT login, credential storage, token refresh, and model access.

You do not need an OpenAI API key. The package does not copy another Codex installation's `auth.json`, implement its own OAuth client, or call an unpublished ChatGPT endpoint.

## Compatibility

This fork keeps the original ChatGPT OAuth/Codex integration and adds compatibility with the DSH 0.2 tool-message contract.

- DSH 0.2 uses first-class `role: "tool"` result messages with `toolCallId` on the message.
- Older DSH releases nested a `tool-result` block inside a user message.
- The adapter accepts both formats and preserves the exact Codex call id when returning tool output.
- The adapter no longer imports the generation-specific `CallId` runtime constructor, which was renamed in the DSH 0.2 line.

The primary target tested by this fork is **DeepSeek Harness 0.2.0-rc.2** while retaining the older result shape as a compatibility fallback.

## Requirements

- Node.js 22.19 or newer
- DeepSeek Harness developer preview (including `0.2.0-rc.2`)
- A ChatGPT account with Codex access

## Install

After this compatibility branch is merged, install directly from this fork with:

```sh
npx -y github:unsocialz2/dsh-openai-oauth install
```

For testing the compatibility branch before merge:

```sh
npx -y github:unsocialz2/dsh-openai-oauth#compat/dsh-0.2.0-rc.2 install
```

The installer registers the package with the `web` and `headless` profiles, plus any custom profiles that already exist. It also replaces the old `deepseek-harness-openai-oauth` and `dsh-llm-codex-app-server` packages if found.

Run the same command to update the package later. If you create a custom profile after installation, run it once more to register that profile.

## Sign in with ChatGPT

Start the Harness web interface:

```sh
npx @deepseek-ai/dsh web
```

Open **Settings > OpenAI OAuth**, select **Sign in with ChatGPT**, and finish authorization in the OpenAI page. The GPT models available to your account will then appear in the normal Harness model picker.

For a headless profile, sign in from the terminal:

```sh
npx @deepseek-ai/dsh plugin --profile headless exec dsh-codex-login
```

The login is stored separately under `~/.deepseek-harness/codex`. Codex owns the credentials and refreshes them when needed.

## Select a model

Choose the provider and model in the Harness interface, or set them in `~/.dsh/settings.yaml`:

```yaml
agent-default-model:
  provider: openai-codex
  model: gpt-5.6-sol
  reasoningEffort: high
```

Run Harness as usual:

```sh
npx @deepseek-ai/dsh --profile headless "inspect this repository"
```

The plugin reads the model list from the signed-in Codex account. It does not keep a hardcoded list of GPT versions.

## Uninstall

Remove the package from every current Harness profile while keeping the ChatGPT login for a future reinstall:

```sh
npx -y dsh-openai-oauth uninstall
```

For a clean uninstall that also signs out and deletes `~/.deepseek-harness/codex`:

```sh
npx -y dsh-openai-oauth uninstall --purge-auth
```

These commands also remove registrations left by the old package name. They do not install a permanent global npm package. npm may retain its normal download cache.

## Local development

```sh
npm install
npm test
npm pack --dry-run
```

The regression suite covers the DSH 0.2 tool-role result format, the older nested result format, tool errors, and empty tool output.

This is an independent community plugin. It is not endorsed by DeepSeek or OpenAI.

## License

MIT
