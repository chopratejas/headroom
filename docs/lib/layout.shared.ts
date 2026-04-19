import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { gitConfig } from './shared';

/** Shared layout options used by both home and docs layouts. */
export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: 'Headroom',
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
    links: [
      {
        text: 'Docs',
        url: '/docs',
        active: 'nested-url',
      },
    ],
  };
}
