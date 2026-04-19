import { docs } from 'collections/server';
import { loader } from 'fumadocs-core/source';
import type { InferPageType } from 'fumadocs-core/source';

export const source = loader({
  baseUrl: '/docs',
  source: docs.toFumadocsSource(),
});

type Page = InferPageType<typeof source>;

/** Build the OG image URL for a given page. */
export function getPageImage(page: Page) {
  return {
    url: `/og/docs/${page.slugs.join('/')}/og.png`,
    segments: [...page.slugs, 'og.png'],
  };
}

/** Build the raw-markdown URL for a given page. */
export function getPageMarkdownUrl(page: Page) {
  return {
    url: `/llms.mdx/docs/${page.slugs.join('/')}/content.md`,
    segments: [...page.slugs, 'content.md'],
  };
}

/** Return the full LLM-friendly text representation of a page. */
export async function getLLMText(page: Page): Promise<string> {
  const md = await page.data.getText('processed');
  return `# ${page.data.title}\n\n${page.data.description ?? ''}\n\n${md}`;
}
