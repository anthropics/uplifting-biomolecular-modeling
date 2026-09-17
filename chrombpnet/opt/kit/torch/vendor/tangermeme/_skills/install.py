# install.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Install the bundled tangermeme Agent Skill for Claude Code.

This module exposes the ``tangermeme-install-skills`` console script. It copies
the Agent Skill that ships inside the installed package into a location where
Claude Code discovers skills, so that a coding agent gains tangermeme-specific
guidance (API contracts and footguns) in every project, not just this repository.

Claude Code does not scan Python ``site-packages`` for skills; it only looks in a
small set of fixed roots. The personal root, ``~/.claude/skills/``, makes a skill
available across all of a user's projects, so that is the default destination. The
skill itself is one self-contained directory -- a ``SKILL.md`` router plus a
``references/`` folder of detailed topic files that are read on demand -- bundled
here under ``data/``, so installation is simply copying that directory.

The copy is deliberately opt-in via this command rather than being performed on
import or install, because silently writing into a user's home configuration would
be surprising and invasive.

Run ``tangermeme-install-skills`` to install, or ``--print-path`` to see where the
bundled source lives (useful with the ``CLAUDE_SKILLS_PATH`` environment variable
if you would rather point Claude Code at the package in place than copy it).
"""

import argparse
import shutil
import sys

from pathlib import Path


SKILL_NAME = "tangermeme"


def _bundled_skill_dir() -> Path:
	"""Return the path to the Agent Skill bundled inside the package.

	Returns
	-------
	skill_dir: pathlib.Path
		The ``data/`` directory containing the bundled ``SKILL.md`` and
		``references/``.
	"""

	return Path(__file__).resolve().parent / "data"


def _default_dest() -> Path:
	"""Return the default install destination in the user's personal skill root.

	Returns
	-------
	dest: pathlib.Path
		``~/.claude/skills/tangermeme``.
	"""

	return Path.home() / ".claude" / "skills" / SKILL_NAME


def install_skill(dest: Path | None = None, force: bool = False) -> Path:
	"""Copy the bundled tangermeme skill into a Claude Code skills directory.

	The bundled skill directory is copied wholesale to ``dest``. If ``dest``
	already exists it is left untouched unless ``force`` is set, in which case it
	is removed and replaced so that an upgrade reflects the installed package
	version exactly.

	Parameters
	----------
	dest: pathlib.Path or None, optional
		The directory to install the skill into. If None, defaults to
		``~/.claude/skills/tangermeme``. Default is None.

	force: bool, optional
		Whether to overwrite an existing installation at ``dest``. Default is
		False.

	Returns
	-------
	dest: pathlib.Path
		The directory the skill was installed into.
	"""

	src = _bundled_skill_dir()
	if not (src / "SKILL.md").is_file():
		raise FileNotFoundError(
			"Bundled skill not found at {}. The package may be installed "
			"without its skill data.".format(src))

	if dest is None:
		dest = _default_dest()

	if dest.exists():
		if not force:
			raise FileExistsError(
				"{} already exists. Re-run with --force to overwrite.".format(
					dest))
		shutil.rmtree(dest)

	dest.parent.mkdir(parents=True, exist_ok=True)
	shutil.copytree(src, dest,
		ignore=shutil.ignore_patterns(".ipynb_checkpoints", "__pycache__"))
	return dest


def main(argv: list[str] | None = None) -> int:
	"""Command-line entry point for ``tangermeme-install-skills``.

	Parameters
	----------
	argv: list of str or None, optional
		Arguments to parse. If None, ``sys.argv`` is used. Default is None.

	Returns
	-------
	code: int
		Process exit code; 0 on success and 1 on a handled error.
	"""

	parser = argparse.ArgumentParser(
		prog="tangermeme-install-skills",
		description="Install the tangermeme Agent Skill for Claude Code.")
	parser.add_argument("--dest", type=Path, default=None,
		help="Destination directory (default: ~/.claude/skills/tangermeme).")
	parser.add_argument("--force", action="store_true",
		help="Overwrite an existing installation.")
	parser.add_argument("--print-path", action="store_true",
		help="Print the bundled skill directory and exit without installing.")
	args = parser.parse_args(argv)

	if args.print_path:
		print(_bundled_skill_dir())
		return 0

	try:
		dest = install_skill(dest=args.dest, force=args.force)
	except (FileExistsError, FileNotFoundError) as e:
		print("error: {}".format(e), file=sys.stderr)
		return 1

	print("Installed tangermeme skill to {}".format(dest))
	print("It will be available to Claude Code in your next session.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
