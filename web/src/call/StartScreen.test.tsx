import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { StartScreen } from "./StartScreen";

const noop = () => {};

describe("StartScreen", () => {
  it("blocks Start and shows why while another tab holds the session", () => {
    const html = renderToStaticMarkup(<StartScreen connected={false}
      error="Another client is already connected." onStart={noop} />);
    expect(html).toMatch(/<button[^>]*disabled=""/);
    expect(html).toContain('role="alert"');
    expect(html).toContain("Another client is already connected.");
  });
  it("is ready when connected", () => {
    const html = renderToStaticMarkup(<StartScreen connected error={null} onStart={noop} />);
    expect(html).not.toMatch(/<button[^>]*disabled=""/);
    expect(html).not.toContain('role="alert"');
    expect(html).toContain("Start call");
  });
});
