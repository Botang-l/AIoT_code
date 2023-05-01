# Contributing to USCC AIoT code

## Table of Contents <!-- omit in toc -->
- [Submission Guidelines](#submission-guidelines)
    - [Submitting a Issue](#submitting-a-issue)
    - [Submitting a Merge Request](#submitting-a-merge-request)
- [Development Rules](#development-rules)
- [Documentation](#documentation)
- [Git Branch Naming](#git-branch-naming)
- [Git Commit Message Convention](#git-commit-message-convention)

## Submission Guidelines
### Submitting a Issue
If you want to discuss something related to the USCC AIoT code such as found a bug or missing a feature, it is welcome to [submit a issue](https://github.com/Botang-l/AIoT_code/issues/new). Before you submit a issue, please search the issue track and make sure there is no similar issue that is still in progress or has come to a conclusion. There are several templates in the repository, please choose a suitable one and write the details as much as better to describe your idea.

### Submitting a Merge Request
After making some improvements for USCC AIoT code, you can submit a merge request to merge your contribution to our [develop](https://github.com/Botang-l/AIoT_code/tree/develop) branch. There are several steps before submitting a merge request:

1. Make sure there is no duplicate existing features.
2. Make sure your commits follow our [development rules](#development-rules) and [commit message conventions](#git-commit-message-convention).


## Development Rules
There are two ways for developers to contribute to this project.
- Create a new branch based on the latest version of `develop` if you are a member of the repository. The naming of the new branch should follow [our convention](#git-branch-naming).
- [Fork the repository](https://playlab.computing.ncku.edu.tw:4001/ITH/website/forks/new) to your account and implement your ideas based on the latest `develop` branch. Follow the [Merge Request Guideline](#submitting-a-merge-request) to merge your work into this project.

Whichever method you choose to work on this project, please comply with the rules described in this document and [Developer Guide](docs/DEVELOPER.md) to keep the quality of our codebase.

## Documentation
To keep this project moving forward, it is important to document the whole development information about the project. Developers can realize the development status and technical details from these documents. Remember to update all related documents while making any changes.

- [development information](https://hackmd.io/@linhoward0522/2023AIoT/https%3A%2F%2Fhackmd.io%2FMNjf3sk6QhyMo1THJLeB2Q%3Fview)

## Git Branch Naming
Except for permanent branches (e.g. `main` and `develop`), all temporary branches should be named in the format `<TYPE>/<ISSUE>-<DETAIL>` such as `feat/1-integer-datatype`. If the goal of a branch has not related to any issues, you can either create an issue to describe the detail of your planning or skip the `<ISSUE>-` in the branch name. With this convention, we can understand the purpose of each branch from its name directly.

There are several catagories that are available for `<TYPE>`:
- `feat`
- `bug`
- `docs`

## Git Commit Message Convention
Follow the rules described in [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) to write commit messages. Each commit should contain only one enhancement to keep the changing history easy to trace. The commits should be structured as:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

There are some common commit types:
- `feat`: A new feature.
- `fix`: A bug to be fixed.
- `BREAKING CHANGE`
- `perf`: Improve the performance of a functionality.
- `refactor`: Restructure the implementation of a functionality without changing its behavior.
- `revert`: Revert a previous commit without other changes.
- `chore`: Changes about build tools or project configurations.
- `test`: Add or correct tests.
- `docs`: Update documentations only.