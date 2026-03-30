# CLAUDE.md

## 提交 PR 到 verl-project/verl 的流程

每次提 PR **必须从 upstream/main 建干净分支**，不能从本地 main 建，否则会把私有改动一起带进去。

```bash
# 1. 拉取上游最新代码
git fetch upstream main

# 2. 从 upstream/main 建新分支（不是从本地 main）
git checkout -b fix/xxx upstream/main

# 3. 只改目标文件
git add 目标文件
git commit -m "..."

# 4. 推到自己的 fork
git push origin fix/xxx

# 5. 提 PR
gh pr create --repo verl-project/verl --head khazic:fix/xxx --base main ...
```

提 PR 前用 `git diff upstream/main` 确认只包含目标改动。
