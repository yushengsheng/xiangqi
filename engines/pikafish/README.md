# Bundled Pikafish

This directory contains the official unmodified Pikafish macOS universal binary and NNUE network.

- Project: https://github.com/official-pikafish/Pikafish
- Release: `Pikafish-2026-09-06`
- Release URL: https://github.com/official-pikafish/Pikafish/releases/tag/Pikafish-2026-09-06
- Download asset: `Pikafish.2026-09-06.7z`
- Binary architectures: macOS `arm64` + `x86_64`
- License: GNU GPL v3 (`COPYING.txt`)
- NNUE licensing: see `NNUE-License.md`

SHA-256:

```text
3a11f9034ef723bb4068e4cf88a79a12d99507474ba118fa916ec2cecd4f4abe  pikafish
7d13d73569a9b571ba0eb20cf1596247bc2a42738967e61afef6482b231e900e  pikafish.nnue
```

The engine is started as a separate UCI subprocess. It is not modified or linked into the Python application.
