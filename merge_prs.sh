#!/bin/bash

# Fetch PRs
git fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*'

# Find all PR branches
PR_BRANCHES=$(git for-each-ref --format='%(refname:short)' refs/remotes/origin/pr/)

for BR in $PR_BRANCHES; do
    PR_MSG=$(git log -1 --format="%s" $BR)
    echo "======================================"
    echo "Attempting to merge: $BR - $PR_MSG"
    
    # Try merging
    if git merge --no-ff -m "Merge $BR" $BR; then
        echo "Merge succeeded. Running tests..."
        
        # Check Rust
        if ~/.cargo/bin/cargo check --workspace --all-targets && ~/.cargo/bin/cargo test --workspace; then
            echo "Tests passed! Keeping the merge."
        else
            echo "Tests failed. Aborting merge."
            git reset --hard HEAD^
        fi
    else
        echo "Merge conflict. Aborting merge."
        git merge --abort
    fi
done

echo "Done."
