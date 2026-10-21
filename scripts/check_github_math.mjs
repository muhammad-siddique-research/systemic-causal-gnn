/**
 * Validate mathematical markup with KaTeX, an additional strict syntax
 * check (GitHub itself renders with MathJax).
 */
import { readFileSync } from "node:fs";
import katex from "katex";

const markdownFiles = [
  "README.md",
  "docs/RESEARCH_STUDY.md",
  "docs/DATA_AND_IDENTIFICATION.md",
  "docs/HEALTH_ECONOMICS_TRANSLATION.md",
  "docs/OBSERVED_MACRO_RESEARCH.md",
];
let count = 0;
const errors = [];
for (const path of markdownFiles) {
  const original = readFileSync(path, "utf8");
  const content = original.replace(/^(\x60{3,}|~{3,})[^\n]*\n[\s\S]*?^\1\s*$/gm, "");
  if (/\\\(|\\\)/.test(content)) {
    errors.push(path + ": old inline \\( ... \\) delimiters remain");
  }
  const blocks = [];
  let working = content.replace(/^\$\$\s*\n([\s\S]*?)^\$\$\s*$/gm, (_, expression) => {
    blocks.push(expression);
    return "";
  });
  const markers = (content.match(/^\$\$\s*$/gm) || []).length;
  if (markers !== blocks.length * 2) {
    errors.push(path + ": unmatched or non-standalone display math delimiter");
  }
  // Ignore Markdown code spans before reading inline mathematical $ ... $.
  working = working.replace(/\x60[^\x60\n]*\x60/g, "");
  const inline = [];
  working.replace(/\$(?!\$)([^$\r\n]+?)\$/g, (_, expression) => {
    inline.push(expression);
    return "";
  });
  for (const [mode, expressions] of [["display", blocks], ["inline", inline]]) {
    for (const [index, expression] of expressions.entries()) {
      try {
        katex.renderToString(expression, {
          displayMode: mode === "display",
          throwOnError: true,
          strict: "ignore",
          trust: false,
        });
        count++;
      } catch (err) {
        errors.push(path + ": " + mode + " formula " + (index + 1)
          + ": " + err.message + "\n" + expression.slice(0, 260));
      }
    }
  }
  console.log(path + ": " + blocks.length + " display and "
    + inline.length + " inline expressions");
}
if (errors.length) {
  console.error("Formula syntax failed:\n" + errors.join("\n\n"));
  process.exitCode = 1;
} else {
  console.log("All " + count + " mathematical expressions parsed successfully.");
}
