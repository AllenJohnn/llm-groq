FROM node:22-bookworm-slim

WORKDIR /app

# Puppeteer is used by development tooling, but its browser is not needed in
# the runtime container. PeerJS is currently a dev dependency and is needed
# by the signaling service, so install the complete lockfile.
ENV PUPPETEER_SKIP_DOWNLOAD=true
COPY package.json package-lock.json ./
RUN npm ci --include=dev

COPY . .

ENV NODE_ENV=production
ENV PORT=8080
ENV SIGNAL_PORT=9000

EXPOSE 8080 9000

CMD ["node", "scripts/container.mjs"]
