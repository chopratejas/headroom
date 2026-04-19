interface CommunityStats {
  total_tokens_saved: number;
  total_cost_saved: number;
  total_requests: number;
  unique_instances: number;
}

const FALLBACK: CommunityStats = {
  total_tokens_saved: 60_000_000_000,
  total_cost_saved: 1_200_000,
  total_requests: 50_000_000,
  unique_instances: 12_000,
};

/** Fetch community stats from Supabase, falling back to static data. */
export async function fetchCommunityStats(): Promise<CommunityStats> {
  const url = process.env.SUPABASE_STATS_URL;
  if (!url) return FALLBACK;
  try {
    const res = await fetch(url, { next: { revalidate: 3600 } });
    if (!res.ok) return FALLBACK;
    return (await res.json()) as CommunityStats;
  } catch {
    return FALLBACK;
  }
}

/** Format a number with K/M/B suffixes. */
export function fmtNum(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toString();
}

/** Format a number as USD with K/M suffixes. */
export function fmtUsd(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
  return `$${n.toFixed(0)}`;
}
