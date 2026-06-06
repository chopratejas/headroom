import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

describe("openclaw plugin manifest contracts", () => {
  it("declares the Headroom retrieval tool registered at runtime", () => {
    const manifest = JSON.parse(
      readFileSync(new URL("../openclaw.plugin.json", import.meta.url), "utf8"),
    );

    expect(manifest.contracts?.tools).toContain("headroom_retrieve");
  });
});
