/**
 * Verify GitHub-safe math syntax rather than only bare LaTeX syntax.
 *
 * - Display equations MUST use fenced "math" blocks (GitHub/MathJax).
 * - Inline equations MUST use protected $ + backtick ... backtick + $.
 * - Parse the extracted TeX through KaTeX with strict error reporting.
 * KaTeX parsing is a supplementary syntax gate, not a screenshot of GitHub.
 */
import { readFileSync } from "node:fs";
import katex from "katex";

const files = [
  "README.md",
  "docs/RESEARCH_STUDY.md",
  "docs/DATA_AND_IDENTIFICATION.md",
  "docs/HEALTH_ECONOMICS_TRANSLATION.md",
  "docs/OBSERVED_MACRO_RESEARCH.md",
];
let overall = 0;
const errors = [];

for (const file of files) {
  const content = readFileSync(file, "utf8");
  const display = [];

  // Separate all Markdown fences before inspecting prose. A math fence
  // retains its complete TeX; source/Python/Mermaid fences are ignored.
  let prose = content.replace(
    /^([ \t]*)(\x60{3,}|~{3,})([^\n]*)\n([\s\S]*?)^\1\2[ \t]*$/gm,
    (_, indent, fence, info, body) => {
      if (info.trim().toLowerCase() === "math") {
        display.push(body);
      }
      return "\n";
    }
  );

  if (/^\$\$\s*$/m.test(prose) || /\\\(|\\\)|\\\[|\\\]/.test(prose)) {
    errors.push(file + ": unprotected display or inline math delimiters");
  }

  const inline = [];
  prose = prose.replace(/\$\x60([^\x60\r\n]+)\x60\$/g, (_, expr) => {
    inline.push(expr);
    return " <protected-math> ";
  });
  // Ignore ordinary Markdown code spans and escaped dollar signs.
  prose = prose.replace(/\x60[^\x60\n]*\x60/g, "");
  prose = prose.replace(/\\\$/g, "");

  // Reject stray math delimiters: KaTeX success would otherwise miss
  // formulas that GitHub parsed as normal Markdown.
  if (prose.includes("$")) {
    errors.push(file + ": unprotected dollar sign/mathematical expression");
  }

  for (const [kind, expressions] of [["display", display], ["inline", inline]]) {
    for (const [index, tex] of expressions.entries()) {
      try {
        katex.renderToString(tex, {
          displayMode: kind === "display",
          throwOnError: true,
          strict: "ignore",
          trust: false,
        });
        overall++;
      } catch (error) {
        errors.push(file + " " + kind + " " + (index + 1) + ": " + error.message);
      }
    }
  }
  console.log(file + ": " + display.length + " protected display and "
    + inline.length + " protected inline equations");
}
if (overall !== 70) {
  errors.push("Expected 70 protected TeX expressions; found " + overall);
}
if (errors.length) {
  console.error("GitHub mathematical markup verification failed:\n"
    + errors.join("\n"));
  process.exitCode = 1;
} else {
  console.log("PASS: all " + overall
    + " equations protected from Markdown and parsed successfully.");
}
