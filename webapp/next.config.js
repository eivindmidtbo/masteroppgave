/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: true,
    serverComponentsExternalPackages: ["moongoose", "@typegoose/typegoose"],
  },
}

module.exports = nextConfig
