#!/usr/bin/env bash
# Builds a shadow copy of the Xcode Command Line Tools that omits the stale
# usr/include/swift/module.modulemap (it duplicates bridging.modulemap and makes
# `xcrun swiftc` fail with "redefinition of module 'SwiftBridging'"). REA compiles a
# small Swift helper for macOS process-ownership checks, so without this its
# JavaScript analyzer cannot start on an affected host. Nothing under /Library is touched.
#
#   scripts/make-shadow-clt.sh [DEST]          # default DEST=/tmp/rea-shadow-clt
#   export REA_DARWIN_SWIFTC_PATH="$DEST/usr/bin/swiftc"
#   export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
#   npx -y rea-agents@6.5.0 analyze-javascript-application <path> --format json
#
# The permanent fix is Apple's: reinstall the Command Line Tools, or remove the
# stale module.modulemap with administrator rights.
set -euo pipefail
CLT=/Library/Developer/CommandLineTools
S=${1:-/tmp/rea-shadow-clt}
[ -d "$CLT" ] || { echo "Command Line Tools not found at $CLT" >&2; exit 1; }
rm -rf "$S"; mkdir -p "$S/usr/bin" "$S/usr/include/swift"
for d in lib libexec share local; do [ -e "$CLT/usr/$d" ] && ln -s "$CLT/usr/$d" "$S/usr/$d"; done
ln -s "$CLT/SDKs" "$S/SDKs"; [ -e "$CLT/Library" ] && ln -s "$CLT/Library" "$S/Library"
for f in "$CLT"/usr/include/*; do b=$(basename "$f"); [ "$b" = swift ] || ln -s "$f" "$S/usr/include/$b"; done
for g in "$CLT"/usr/include/swift/*; do [ "$(basename "$g")" = module.modulemap ] || ln -s "$g" "$S/usr/include/swift/"; done
# swiftc and swift-frontend must be real copies: the driver derives its toolchain root from its own real path.
for b in "$CLT"/usr/bin/*; do n=$(basename "$b"); case $n in
  swiftc|swift-frontend|swift-driver|swift) cp -R "$(readlink -f "$b")" "$S/usr/bin/$n";;
  *) ln -s "$b" "$S/usr/bin/$n";; esac; done
printf 'import Foundation\nprint("ok")\n' > "$S/t.swift"
SDKROOT="$CLT/SDKs/MacOSX.sdk" "$S/usr/bin/swiftc" -sdk "$CLT/SDKs/MacOSX.sdk" "$S/t.swift" -o "$S/t" && "$S/t" && rm -f "$S/t" "$S/t.swift"
echo "shadow toolchain ready: export REA_DARWIN_SWIFTC_PATH=$S/usr/bin/swiftc SDKROOT=$CLT/SDKs/MacOSX.sdk"
