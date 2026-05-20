# VHS tape — generates docs/demo.gif for the README.
#
# Run with:  vhs docs/demo.tape
# Install:   brew install vhs    (https://github.com/charmbracelet/vhs)
#
# The demo runs through `redforge init` + `redforge scan --dry-run`. No API key
# required because --dry-run never calls the target or the judge. The whole run
# fits in ~15 seconds.

Output docs/demo.gif

Set FontSize 16
Set Width 1280
Set Height 720
Set Theme "Catppuccin Mocha"
Set TypingSpeed 50ms
Set PlaybackSpeed 1.2
Set Margin 24
Set MarginFill "#1e1e2e"
Set BorderRadius 8
Set Shell "bash"

# Hidden setup: activate the project venv and set a clean prompt.
Hide
Type "export PATH=/Users/danielblanco/Documents/Documents/Redforge/.venv/bin:$PATH"
Enter
Sleep 200ms
Type "export PS1='$ '"
Enter
Sleep 200ms
Type "rm -rf /tmp/redforge-demo && mkdir -p /tmp/redforge-demo && cd /tmp/redforge-demo && clear"
Enter
Sleep 500ms
Show

# Scaffold a new project.
Type "redforge init"
Sleep 500ms
Enter
Sleep 2s

# Show the scaffolded files.
Type "ls"
Sleep 200ms
Enter
Sleep 1500ms

# Preview a scan without calling the target or judge.
Type "redforge scan --dry-run"
Sleep 200ms
Enter
Sleep 4s

# End on the dry-run summary.
Sleep 1500ms
