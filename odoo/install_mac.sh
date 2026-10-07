#!/bin/bash
# ONE-TIME SETUP, run by whoever installs the Mac (not the daily user):
#
#     bash install_mac.sh
#
# Run it with "bash" in front like this and nothing else needs to be made
# executable or approved. It sets the whole system up, then puts two icons on
# the Desktop: "Bakery Invoices" (start and open) and "Stop Bakery". After
# this, nobody ever needs the Terminal again.

cd "$(dirname "$0")" || exit 1
HERE="$(pwd)"
export PATH="/usr/local/bin:/opt/homebrew/bin:/Applications/Docker.app/Contents/Resources/bin:$PATH"

chmod +x ./*.command

echo "Setting up the invoice system. The first time this downloads a lot and"
echo "takes 10-15 minutes. Please leave this window open until it says DONE."
echo

./"Start Bakery.command" --quiet || { echo; echo "Setup did not finish - see the message above."; exit 1; }

DESKTOP="$HOME/Desktop"
TMP="$(mktemp -d)"

# The icons are small AppleScript programs built here, on this Mac, so macOS
# treats them as the user's own (a downloaded program would instead trigger an
# "unidentified developer" warning the first time).
cat > "$TMP/start.applescript" <<EOF
set startScript to "$HERE/Start Bakery.command"
display notification "Starting - this can take a minute or two." with title "Bakery Invoices"
try
	with timeout of 900 seconds
		do shell script (quoted form of startScript) & " --quiet 2>&1"
	end timeout
on error errMsg
	display dialog "The invoice system could not start." & return & return & errMsg buttons {"OK"} default button "OK" with icon stop
	return
end try
open location "http://localhost:8069"
EOF

cat > "$TMP/stop.applescript" <<EOF
set stopScript to "$HERE/Stop Bakery.command"
display notification "Saving a backup and stopping..." with title "Bakery Invoices"
try
	with timeout of 300 seconds
		do shell script (quoted form of stopScript) & " 2>&1"
	end timeout
on error errMsg
	display dialog "Could not stop cleanly." & return & return & errMsg buttons {"OK"} default button "OK" with icon stop
	return
end try
display notification "Stopped. Everything is saved." with title "Bakery Invoices"
EOF

rm -rf "$DESKTOP/Bakery Invoices.app" "$DESKTOP/Stop Bakery.app"
osacompile -o "$DESKTOP/Bakery Invoices.app" "$TMP/start.applescript" || { echo "Could not create the Desktop icon."; exit 1; }
osacompile -o "$DESKTOP/Stop Bakery.app" "$TMP/stop.applescript" || { echo "Could not create the Stop icon."; exit 1; }
rm -rf "$TMP"

echo
echo "DONE."
echo "Two icons are now on the Desktop: 'Bakery Invoices' and 'Stop Bakery'."
echo "Still to do by hand (see README_MAC.md, setup day): make Docker Desktop"
echo "start when the Mac starts, and log in once in the browser and let it remember the password."
[ -f "$HERE/Bakery login.txt" ] && { echo; cat "$HERE/Bakery login.txt"; }
