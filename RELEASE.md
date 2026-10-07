# Releases

LLM ShardX deploys through Vercel's git integration. Git tags mark milestones people can cite.

| Push to | Deploys to |
|---|---|
| any branch | Preview deployment on Vercel |
| `faster-kernels` | also `https://webslice-dev.vercel.app` (the staging name) |
| `main` | production, [webslice.ai](https://webslice.ai) |

So every merge to `main` is a production release: validate on the branch preview first, and merge with a pull request.

## Cutting a release

1. All GPU tests green on the maintainer's hardware (`npm run test:gpu`, `npm run test:q38`).
2. [docs/bench-log.md](docs/bench-log.md) has rows for every performance change since the last tag.
3. Update [CHANGELOG.md](CHANGELOG.md): move *Unreleased* into a dated section.
4. Tag: `git tag -a v0.X.0 -m "..." && git push --tags`.
5. Merge to `main`; Vercel deploys production automatically (a manual `npx vercel deploy --prod` is only for emergencies).

Versioning is `0.MINOR.PATCH` until the peer protocol is declared stable; a protocol change bumps MINOR.
