import platform
import warnings
from datetime import date

from pytorch_lightning import Callback

try:
    from git.repo import Repo as GitRepo
    from git import IndexFile
    _GIT_AVAILABLE = True
except Exception as _git_import_err:
    _GIT_AVAILABLE = False
    _git_import_err_msg = str(_git_import_err)


def commit_cwd(branch, message, repo=None):
    repo = repo or GitRepo('.')
    index = IndexFile(repo)
    log_branch = getattr(repo.heads, branch, False) or repo.create_head(branch)

    to_add = (
        set(repo.untracked_files)
        | set([d.a_path or d.b_path for d in index.diff(None) if d.change_type != 'D'])
        | set([d.a_path or d.b_path for d in index.diff(repo.head.commit) if d.change_type != 'A'])
    ) - set([d.a_path or d.b_path for d in index.diff(None) if d.change_type == 'D'])

    index.add(
        to_add,
        force=False,
        write=False,
        write_extension_data=True,
    )

    log_commit = index.commit(message, parent_commits=[log_branch.commit], head=False)
    log_branch.commit = log_commit
    return log_commit


class VersioningCallback(Callback):
    def __init__(self, repo_path='.', branch='run_log'):
        self.setup_hash = None
        self.date = str(date.today())
        self.branch = f'{branch}-{self.date}-{platform.node()}'
        self.git_repo = None
        if not _GIT_AVAILABLE:
            warnings.warn(f'VersioningCallback: git unavailable ({_git_import_err_msg}), skipping versioning.')
            return
        try:
            self.git_repo = GitRepo(repo_path)
        except Exception as e:
            warnings.warn(f'VersioningCallback: could not open git repo ({e}), skipping versioning.')

    def setup(self, trainer, pl_module, stage):
        if self.git_repo is None:
            return
        try:
            msg = ''
            if trainer.logger is not None:
                msg = trainer.logger.log_dir
            commit = commit_cwd(
                self.branch,
                f'Setup {stage} {msg}',
                repo=self.git_repo,
            )
            self.setup_hash = str(commit)
        except Exception as e:
            warnings.warn(f'VersioningCallback: git commit failed ({e}), skipping versioning.')
