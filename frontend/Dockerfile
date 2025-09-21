# Build stage
FROM node:18-alpine AS build

# Install Chromium for remote debugging
RUN apk add --no-cache chromium

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# For development
ENV CHROME_BIN=/usr/bin/chromium-browser
ENV CHROME_PATH=/usr/lib/chromium/

CMD ["npm", "run", "dev"] 