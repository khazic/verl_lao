# Git Merge Safety

## Incident Summary

On April 7, 2026, branch `lao` was not lost, but a bad merge commit rewrote the branch tree into an almost empty state.

- Normal parent tree size: about 1198 tracked files
- `upstream/main` parent tree size: about 1163 tracked files
- Bad merge commit `3dcb0a09`: 2 tracked files
- Follow-up commit `c810b0ba`: 3 tracked files

The failure point was the merge commit itself. The server only reflected the broken remote history.

## Likely Cause

The most likely cause was a broken or overwritten Git index during conflict resolution or merge finalization. If the index only contains a few paths, `git commit` will write a tree with only those paths, even if many files still exist in the working directory.

Common danger patterns:

- `.git/index` size changes abnormally or backup-like files appear
- most files suddenly become `??`
- merge result shows massive deletions that were not intended
- only conflict files are re-added, while the rest of the tree is missing from the index

## Mandatory Checks Before Any Merge Commit

Run these checks before `git commit` after a merge or conflict resolution:

```bash
git status --short --branch
git diff --cached --stat
git ls-files | wc -l
git diff --name-status HEAD
```

If the repo normally has about 1000+ tracked files and `git ls-files | wc -l` suddenly drops to a tiny number, stop immediately. Do not commit.

## Mandatory Checks After Any Merge Commit

Immediately verify the produced commit:

```bash
git show --stat --summary --format=fuller HEAD
git ls-tree -r --name-only HEAD | wc -l
git diff --name-status HEAD^ HEAD | sed -n '1,50p'
```

If you see mass deletions or an implausibly small tree, treat the merge as broken and fix it before pushing.

## Fail-Closed Rule

Stop and investigate if any of the following happens:

- almost all files appear as untracked
- tracked file count collapses unexpectedly
- merge commit deletes large parts of the repo unexpectedly
- `.git/index` looks suspiciously small

In that situation:

```bash
git branch backup/<name>-broken-<date> HEAD
git log --oneline --decorate -n 10
git show --stat --summary HEAD
```

Do not push until the tree size and diff both look reasonable.
