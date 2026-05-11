---
description: Ensure safe Git operations when reverting changes to avoid breaking commit histories or causing merge conflicts.
---

# Safe Git Operations & Reverting Changes

When a user asks to undo, revert, or remove changes, you MUST adhere to the following rules regarding Git operations:

- **Never Rewrite History:** Do not use destructive or history-altering commands like `git reset --hard`, `git push -f` (force push), or interactive rebases (`git rebase -i`), especially on branches that exist remotely.
- **Avoid Merge Conflicts:** Rewriting history can cause severe merge conflicts for the user later. Prioritize operations that preserve the existing commit timeline.
- **Safe Reverts Only:** If an applied code change needs to be undone:
  - Prefer using string replacement tools or file edits to manually reverse the code.
  - If using Git to rollback files, use `git restore <file>` or `git checkout -- <file>` to undo uncommitted changes.
  - If a change is already committed, use `git revert <commit-hash>` to create a forward-moving commit that undoes the changes safely.
- **Respect Checkpoints:** Rely on workspace states or safe file restorations instead of trying to artificially bend the Git tree.