# Use an official Node.js runtime as a parent image
FROM node:20.17.0 AS development

ARG API_URL
RUN echo "Building with API URL: $API_URL"
# Set the working directory inside the container
WORKDIR /usr/src/app


# install app dependencies
COPY ./package.json ./
COPY ./package-lock.json ./

# Install app dependencies
RUN npm install

FROM development AS build
# add app
COPY . ./

RUN npm run build

FROM nginx:alpine AS production

WORKDIR /usr/share/nginx/html

COPY --from=build /usr/src/app/dist .
COPY default.conf /etc/nginx/conf.d/default.conf