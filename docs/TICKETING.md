# Ticketing

Use Linear as the source of truth for product and engineering work.

## Workflow

1. Create or pick a Linear issue before starting work.
2. Put the Linear issue key in the branch name, PR title, and commit subject when possible.
3. Keep status in Linear current: Backlog, Todo, In Progress, In Review, Done.
4. Link PRs from Linear or use the Linear GitHub integration so merged PRs close issues.
5. Use GitHub issues only for external reports if the repo becomes public.

## Branches

Use short branch names:

```text
codex/ANA-123-chat-interface
akacolf/ANA-124-training-metrics
```

## Commits

Prefer concise subjects with the issue key first:

```text
ANA-123 Add interactive chat CLI
ANA-124 Report validation accuracy during training
```

## Automation

Enable the Linear GitHub integration for this repo and map:

- PR opened: move issue to In Review.
- PR merged: move issue to Done.
- Commit or PR mention with issue key: link to the issue.

This keeps implementation details in GitHub while Linear remains the planning board.
