import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  basePath: '/headroom',
  images: { unoptimized: true },
  reactStrictMode: true,
  serverExternalPackages: ['typescript', 'twoslash'],
};

export default withMDX(config);
