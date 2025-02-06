FROM oven/bun:alpine AS build

WORKDIR /app

COPY . .

RUN bun install

EXPOSE 3457

CMD ["bun", "run", "prod"]