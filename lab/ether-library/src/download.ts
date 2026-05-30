/**
 * download.ts — fetch release/installer artifacts for the languages in
 * `@ether/library/Language`, for a specific / latest / all version, from
 * wherever the language actually publishes.
 *
 * Not every language has a downloadable release. Surveying the per-language
 * install.sh, sources fall into:
 *
 *   - "github"   — a GitHub repo. We list *releases* first (binary assets,
 *                  filtered to linux/x86-64 when possible); if a repo cuts no
 *                  releases we fall back to git *tags* and the per-tag source
 *                  tarball. Supports latest / all / <tag>.
 *   - "direct"   — a vendor download index. Special-cased for Go (go.dev/dl
 *                  JSON) and Zig (ziglang.org index.json); otherwise a single
 *                  templated URL with {version}.
 *   - "apt"      — distro-only languages. We pull the .deb(s) without
 *                  installing (apt-get download / --print-uris). apt only
 *                  serves the versions its repos hold, so "all" degrades to
 *                  whatever `apt-cache madison` lists (best-effort).
 *   - "registry" — pip / npm / cargo / gem / opam. Best-effort `pip download`
 *                  etc.; "all" is generally not enumerable.
 *   - "none"     — notation / historical / commercial: nothing to download.
 *
 * Each language's Source is inferred from its install.sh (GitHub repo, apt
 * package name, release URL, registry), with a small hand-curated override
 * table for the cases inference gets wrong (Go, Zig, …).
 *
 *   - sourceFor(language)              → Source (curated override or inferred).
 *   - listVersions(source, sel)        → resolved version list.
 *   - download(source, sel, destDir)   → DownloadResult[] (writes files).
 *
 * CLI:
 *   tsx src/download.ts source <lang>            # show the resolved Source
 *   tsx src/download.ts list   <lang> [--all]    # list versions (latest, or all)
 *   tsx src/download.ts get    <lang> [<ver>|latest|all] [--dest DIR] [--deps]
 */

import { spawnSync }     from "node:child_process";
import * as fs           from "node:fs";
import * as os           from "node:os";
import * as path         from "node:path";
import { languageDir, which } from "./elf.ts";

// ── Source model ───────────────────────────────────────────────────────────
export type SourceKind = "github" | "direct" | "apt" | "registry" | "none";

export interface Source {
  language:    string;
  kind:        SourceKind;
  repo?:       string;                 // github "owner/name"
  index?:      "go" | "zig";           // known vendor version index
  urlTemplate?: string;                // direct url, {version} substituted
  pkg?:        string;                 // apt package
  registry?:   "pip" | "npm" | "cargo" | "gem" | "opam";
  reason?:     string;                 // why "none"
  inferred?:   boolean;
}

export type Selector = "latest" | "all" | string;   // string = explicit version/tag

export interface Version {
  version:  string;
  /** download URLs for this version (assets / tarball / package uris). */
  urls:     string[];
  kind:     "release-asset" | "source-tarball" | "direct" | "deb" | "registry";
}

// ── Curated overrides (where inference is wrong or a vendor index exists) ───
const OVERRIDES: Record<string, Partial<Source>> = {
  Go:         { kind: "direct", index: "go" },
  Zig:        { kind: "direct", index: "zig" },
  Rust:       { kind: "github", repo: "rust-lang/rust" },     // rustup installs; tags are the version source
  Crystal:    { kind: "github", repo: "crystal-lang/crystal" },
  Node:       { kind: "direct", urlTemplate: "https://nodejs.org/dist/{version}/node-{version}-linux-x64.tar.xz" },
  Deno:       { kind: "github", repo: "denoland/deno" },
  Bun:        { kind: "github", repo: "oven-sh/bun" },
};

// ── Inference from install.sh ───────────────────────────────────────────────
function readInstall(language: string): string | null {
  const dir = languageDir();
  if (!dir) return null;
  const sh = path.join(dir, language, "install.sh");
  return fs.existsSync(sh) ? fs.readFileSync(sh, "utf8") : null;
}

function inferGithubRepo(text: string): string | undefined {
  // Skip raw/gist hosts; take the first clone/release repo reference.
  const m = text.match(/github\.com\/([A-Za-z0-9_.-]+)\/([A-Za-z0-9_.-]+?)(?:\.git|["'\s/)]|$)/);
  if (!m) return undefined;
  if (/^(raw|gist|api|codeload)$/.test(m[1])) return undefined;
  return `${m[1]}/${m[2]}`;
}

function inferAptPkg(text: string): string | undefined {
  const m = text.match(/apt-get\s+install\s+(?:-y\s+|--[\w-]+(?:=\S+)?\s+)*([A-Za-z0-9][\w.+-]*)/);
  return m ? m[1] : undefined;
}

function inferRegistry(text: string): Source["registry"] | undefined {
  if (/\bpip3?\s+install\b/.test(text))        return "pip";
  if (/\bnpm\s+install\b/.test(text))          return "npm";
  if (/\bcargo\s+install\b/.test(text))        return "cargo";
  if (/\bgem\s+install\b/.test(text))          return "gem";
  if (/\bopam\s+install\b/.test(text))         return "opam";
  return undefined;
}

function inferReleaseUrl(text: string): string | undefined {
  const m = text.match(/https?:\/\/\S+?(?:releases\/download|\/dl\/)\S+/);
  return m ? m[0].replace(/["')]+$/, "") : undefined;
}

export function sourceFor(language: string): Source {
  const base: Source = { language, kind: "none", inferred: true };
  const ov = OVERRIDES[language];
  if (ov) return { language, inferred: false, kind: "github", ...ov } as Source;

  const text = readInstall(language);
  if (text === null) return { ...base, reason: "no install.sh" };

  // Historical / unavailable markers.
  if (/\bexit 1\b/.test(text) && /(historical|commercial|no (standard|modern|installation)|No installation required|notation)/i.test(text))
    return { ...base, reason: "no installable artifact (historical/notation/commercial)" };

  const repo = inferGithubRepo(text);
  const relUrl = inferReleaseUrl(text);
  const apt = inferAptPkg(text);
  const reg = inferRegistry(text);

  if (relUrl && !repo) return { ...base, kind: "direct", urlTemplate: relUrl };
  if (repo)            return { ...base, kind: "github", repo };
  if (apt)             return { ...base, kind: "apt", pkg: apt };
  if (reg)             return { ...base, kind: "registry", registry: reg };
  return { ...base, reason: "no recognizable source in install.sh" };
}

// ── HTTP helpers ─────────────────────────────────────────────────────────
async function getJson(url: string): Promise<any> {
  const headers: Record<string, string> = { "User-Agent": "ether-library" };
  if (process.env.GITHUB_TOKEN && url.includes("api.github.com"))
    headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  const r = await fetch(url, { headers });
  if (!r.ok) throw new Error(`GET ${url} → ${r.status} ${r.statusText}`);
  return r.json();
}

const LINUX_X64 = /(linux|unknown-linux).*(x86[_-]?64|amd64|x64)|(x86[_-]?64|amd64|x64).*(linux)/i;
// Non-binary release assets that happen to match the arch filter.
const NOT_A_BINARY = /profile|sbom|\.(sha\d*|asc|sig|pem|txt|json|md)$/i;

// ── Version listing ─────────────────────────────────────────────────────
async function githubReleases(repo: string): Promise<Version[]> {
  const out: Version[] = [];
  for (let page = 1; page <= 10; page++) {
    const rels: any[] = await getJson(`https://api.github.com/repos/${repo}/releases?per_page=100&page=${page}`);
    if (!rels.length) break;
    for (const rel of rels) {
      const assets = (rel.assets ?? []).filter((a: any) => LINUX_X64.test(a.name) && !NOT_A_BINARY.test(a.name));
      out.push({
        version: rel.tag_name,
        kind: assets.length ? "release-asset" : "source-tarball",
        urls: assets.length ? assets.map((a: any) => a.browser_download_url) : [rel.tarball_url],
      });
    }
    if (rels.length < 100) break;
  }
  return out;
}

async function githubTags(repo: string): Promise<Version[]> {
  const out: Version[] = [];
  for (let page = 1; page <= 10; page++) {
    const tags: any[] = await getJson(`https://api.github.com/repos/${repo}/tags?per_page=100&page=${page}`);
    if (!tags.length) break;
    for (const t of tags)
      out.push({ version: t.name, kind: "source-tarball", urls: [`https://github.com/${repo}/archive/refs/tags/${t.name}.tar.gz`] });
    if (tags.length < 100) break;
  }
  return out;
}

async function goVersions(): Promise<Version[]> {
  const list: any[] = await getJson("https://go.dev/dl/?mode=json&include=all");
  return list.map((v: any) => {
    const f = (v.files ?? []).find((f: any) => f.os === "linux" && f.arch === "amd64" && f.kind === "archive");
    return { version: v.version, kind: "direct" as const, urls: f ? [`https://go.dev/dl/${f.filename}`] : [] };
  }).filter(v => v.urls.length);
}

async function zigVersions(): Promise<Version[]> {
  const idx: any = await getJson("https://ziglang.org/download/index.json");
  const out: Version[] = [];
  for (const [version, info] of Object.entries<any>(idx)) {
    const t = info["x86_64-linux"];
    if (t?.tarball) out.push({ version, kind: "direct", urls: [t.tarball] });
  }
  // Prefer tagged stable releases; push the rolling "master" build to the end.
  return out.sort((a, b) => Number(a.version === "master") - Number(b.version === "master"));
}

function aptMadison(pkg: string): Version[] {
  const r = spawnSync("apt-cache", ["madison", pkg], { encoding: "utf8" });
  if (r.status !== 0) return [];
  const out: Version[] = [];
  for (const line of r.stdout.split("\n")) {
    const m = line.split("|").map(s => s.trim());
    if (m.length >= 2 && m[0] === pkg) out.push({ version: m[1], kind: "deb", urls: [] });
  }
  return out;
}

export async function listVersions(source: Source, sel: Selector = "latest"): Promise<Version[]> {
  let all: Version[] = [];
  switch (source.kind) {
    case "github": {
      all = await githubReleases(source.repo!);
      if (!all.length) all = await githubTags(source.repo!);
      break;
    }
    case "direct":
      if (source.index === "go")  all = await goVersions();
      else if (source.index === "zig") all = await zigVersions();
      else if (source.urlTemplate) all = [{ version: "as-published", kind: "direct", urls: [source.urlTemplate] }];
      break;
    case "apt":
      all = aptMadison(source.pkg!);
      break;
    case "registry":
    case "none":
      all = [];
      break;
  }

  if (sel === "all")    return all;
  if (sel === "latest") return all.slice(0, 1);
  const hit = all.filter(v => v.version === sel || v.version === `v${sel}` || v.version.replace(/^v/, "") === sel);
  return hit.length ? hit : [{ version: sel, kind: source.kind === "apt" ? "deb" : "source-tarball", urls: [] }];
}

// ── Download ─────────────────────────────────────────────────────────────
export interface DownloadResult {
  version: string;
  files:   string[];     // local paths written
  ok:      boolean;
  error?:  string;
}

async function fetchTo(url: string, dest: string): Promise<void> {
  const r = await fetch(url, { headers: { "User-Agent": "ether-library" } });
  if (!r.ok) throw new Error(`GET ${url} → ${r.status}`);
  const buf = Buffer.from(await r.arrayBuffer());
  fs.writeFileSync(dest, buf);
}

function aptDownload(pkg: string, version: string | null, destDir: string, deps: boolean): DownloadResult {
  const tag = version ?? "latest";
  if (deps) {
    // Rootless: ask apt for the URIs of pkg + deps, then fetch them ourselves.
    const r = spawnSync("apt-get", ["--print-uris", "install", "-y", version ? `${pkg}=${version}` : pkg],
      { encoding: "utf8" });
    if (r.status !== 0) return { version: tag, files: [], ok: false, error: r.stderr.trim() || "apt-get --print-uris failed" };
    const files: string[] = [];
    for (const line of r.stdout.split("\n")) {
      const m = line.match(/^'([^']+)'\s+(\S+)/);
      if (!m) continue;
      const [, url, name] = m;
      const out = path.join(destDir, name);
      const c = spawnSync("curl", ["-fsSL", "-o", out, url]);
      if (c.status === 0) files.push(out);
    }
    return { version: tag, files, ok: files.length > 0, error: files.length ? undefined : "no .deb URIs resolved" };
  }
  // Single package, rootless via apt-get download (writes into cwd).
  const r = spawnSync("apt-get", ["download", version ? `${pkg}=${version}` : pkg],
    { encoding: "utf8", cwd: destDir });
  if (r.status !== 0) return { version: tag, files: [], ok: false, error: r.stderr.trim() || "apt-get download failed" };
  const debs = fs.readdirSync(destDir).filter(f => f.endsWith(".deb")).map(f => path.join(destDir, f));
  return { version: tag, files: debs, ok: debs.length > 0 };
}

export async function download(source: Source, sel: Selector, destDir: string, deps = false): Promise<DownloadResult[]> {
  fs.mkdirSync(destDir, { recursive: true });

  if (source.kind === "apt") {
    if (sel === "all") return aptMadison(source.pkg!).map(v => aptDownload(source.pkg!, v.version, destDir, deps));
    const ver = sel === "latest" ? null : sel;
    return [aptDownload(source.pkg!, ver, destDir, deps)];
  }
  if (source.kind === "registry")
    return [{ version: sel, files: [], ok: false, error: `registry (${source.registry}) download not implemented; use ${source.registry} directly` }];
  if (source.kind === "none")
    return [{ version: sel, files: [], ok: false, error: source.reason ?? "no downloadable source" }];

  const versions = await listVersions(source, sel);
  const results: DownloadResult[] = [];
  for (const v of versions) {
    if (!v.urls.length) { results.push({ version: v.version, files: [], ok: false, error: "no resolvable URL" }); continue; }
    const vdir = sel === "all" ? path.join(destDir, v.version) : destDir;
    fs.mkdirSync(vdir, { recursive: true });
    const files: string[] = [];
    let error: string | undefined;
    for (const url of v.urls) {
      const name = path.basename(new URL(url).pathname) || `${source.language}-${v.version}`;
      const out = path.join(vdir, name);
      try { await fetchTo(url, out); files.push(out); }
      catch (e: any) { error = e.message; }
    }
    results.push({ version: v.version, files, ok: files.length > 0, error: files.length ? undefined : error });
  }
  return results;
}

// ── CLI ──────────────────────────────────────────────────────────────────
function fmtSource(s: Source): string {
  const coord =
    s.kind === "github"   ? s.repo :
    s.kind === "direct"   ? (s.index ?? s.urlTemplate) :
    s.kind === "apt"      ? s.pkg :
    s.kind === "registry" ? s.registry :
    s.reason;
  return `${s.language}: ${s.kind}${coord ? ` (${coord})` : ""}${s.inferred ? "  [inferred]" : ""}`;
}

async function main() {
  const [cmd, lang, ...rest] = process.argv.slice(2);
  if (!cmd || !lang) {
    console.log("usage: tsx src/download.ts <source <lang> | list <lang> [--all] | get <lang> [<ver>|latest|all] [--dest DIR] [--deps]>");
    return;
  }
  const source = sourceFor(lang);

  if (cmd === "source") { console.log(fmtSource(source)); return; }

  if (cmd === "list") {
    const sel: Selector = rest.includes("--all") ? "all" : "latest";
    const vs = await listVersions(source, sel);
    if (!vs.length) { console.log(`${fmtSource(source)}\n  (no versions resolvable)`); return; }
    console.log(fmtSource(source));
    for (const v of vs) console.log(`  ${v.version.padEnd(20)} ${v.kind}  ${v.urls[0] ?? ""}`);
    return;
  }

  if (cmd === "get") {
    const di = rest.indexOf("--dest");
    const dest = di >= 0 ? rest[di + 1] : path.join(os.tmpdir(), `ether-dl-${lang}`);
    const deps = rest.includes("--deps");
    const sel: Selector = rest.find(a => !a.startsWith("--") && a !== dest) ?? "latest";
    console.log(`${fmtSource(source)}\n  → ${dest}`);
    const results = await download(source, sel, dest, deps);
    for (const r of results) {
      console.log(r.ok ? `  OK   ${r.version}: ${r.files.map(f => path.basename(f)).join(", ")}`
                       : `  FAIL ${r.version}: ${r.error}`);
    }
    return;
  }

  console.error(`unknown command: ${cmd}`);
  process.exit(1);
}

if (process.argv[1] && import.meta.url === `file://${path.resolve(process.argv[1])}`) {
  main().catch(e => { console.error(e.message); process.exit(1); });
}
