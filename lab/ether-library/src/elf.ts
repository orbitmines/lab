/**
 * elf.ts — produce x86-64 ELF executables from a "hello world" for as many of
 * the languages in `@ether/library/Language` as we have a path to native code.
 *
 * A *strategy* is one concrete route from source to an x86-64 ELF. A language
 * can have several (e.g. C via gcc, via clang, via the bundled LLVM clang; or
 * Python via nuitka, via pyinstaller). Strategies are grouped by:
 *
 *   - "native"    — the compiler emits an ELF directly (C, Rust, Go, Zig, …).
 *   - "transpile" — source is lowered to C/LLVM-IR first, then assembled to ELF
 *                   (Vala → C, .ll → clang).
 *   - "aot"       — an ahead-of-time compiler for an otherwise-VM language
 *                   (GraalVM native-image, kotlin-native, .NET NativeAOT).
 *   - "freeze"    — an interpreter + bytecode is bundled into a single ELF
 *                   launcher (PyInstaller, Nuitka --onefile, bun/deno compile).
 *
 * The driver writes the sample, runs the strategy's command steps, then reads
 * the produced file's ELF header and confirms it is ELFCLASS64 / little-endian
 * / EM_X86_64 before reporting success. Toolchain availability is probed up
 * front; missing toolchains map back to the language's own install.sh.
 *
 *   - STRATEGIES                      → the full catalog.
 *   - strategiesFor(language)         → strategies for one language.
 *   - verifyElf(path)                 → { ok, class, endian, machine, type }.
 *   - build(strategy, opts?)          → BuildResult (writes a real ELF).
 *
 * CLI:
 *   tsx src/elf.ts list                 # catalog, grouped by language
 *   tsx src/elf.ts list --available     # only strategies whose toolchain exists
 *   tsx src/elf.ts build C              # build every strategy for C
 *   tsx src/elf.ts build C gcc          # build one strategy by id
 *   tsx src/elf.ts build --all          # attempt everything available, report
 *   tsx src/elf.ts install [<lang>...]  # run each language's own install.sh
 *   tsx src/elf.ts probe                # which toolchains are present
 */

import { spawnSync }       from "node:child_process";
import * as fs             from "node:fs";
import * as os             from "node:os";
import * as path           from "node:path";
import { fileURLToPath }   from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));

// ── Language dir resolution ───────────────────────────────────────────────
// Walk upward from here looking for `ray/@ether/library/Language`, or honour
// $ETHER_LANGUAGE_DIR. This is where each language keeps install/check/run.sh.
export function languageDir(): string | null {
  if (process.env.ETHER_LANGUAGE_DIR) return process.env.ETHER_LANGUAGE_DIR;
  let dir = HERE;
  for (let i = 0; i < 8; i++) {
    const cand = path.join(dir, "ray/@ether/library/Language");
    if (fs.existsSync(cand)) return cand;
    const up = path.dirname(dir);
    if (up === dir) break;
    dir = up;
  }
  return null;
}

function isExecutable(p: string): boolean {
  try { fs.accessSync(p, fs.constants.X_OK); return fs.statSync(p).isFile(); }
  catch { return false; }
}

const whichCache = new Map<string, string | null>();

/** Resolve an executable against $PATH. */
export function which(exe: string): string | null {
  if (whichCache.has(exe)) return whichCache.get(exe)!;
  let found: string | null = null;
  if (exe.includes("/")) {
    found = isExecutable(exe) ? path.resolve(exe) : null;
  } else {
    for (const dir of (process.env.PATH ?? "").split(path.delimiter)) {
      if (!dir) continue;
      const p = path.join(dir, exe);
      if (isExecutable(p)) { found = p; break; }
    }
  }
  whichCache.set(exe, found);
  return found;
}

// ── Strategy model ─────────────────────────────────────────────────────────
export type StrategyKind = "native" | "transpile" | "aot" | "freeze";

export interface Strategy {
  language:   string;     // must match a Language/<dir> name
  id:         string;     // unique within the language, e.g. "gcc"
  kind:       StrategyKind;
  toolchain:  string[];   // executables that must all be present
  sourceName: string;     // filename the sample is written to in the workdir
  sample:     string;     // a minimal hello-world program
  /** argv templates. Tokens: {src} {out} {dir} {base}. {tool:NAME} → resolved path of NAME. */
  steps:      string[][];
  /** template for the produced ELF, if not {out}. */
  artifact?:  string;
  notes?:     string;
}

const HELLO = "hello, ether";

// ── Catalog ──────────────────────────────────────────────────────────────
// Native: the compiler writes an ELF directly.
const NATIVE: Strategy[] = [
  {
    language: "C", id: "gcc", kind: "native", toolchain: ["gcc"],
    sourceName: "hello.c",
    sample: `#include <stdio.h>\nint main(void){puts("${HELLO}");return 0;}\n`,
    steps: [["{tool:gcc}", "{src}", "-o", "{out}"]],
  },
  {
    language: "C", id: "clang", kind: "native", toolchain: ["clang"],
    sourceName: "hello.c",
    sample: `#include <stdio.h>\nint main(void){puts("${HELLO}");return 0;}\n`,
    steps: [["{tool:clang}", "{src}", "-o", "{out}"]],
  },
  {
    language: "C++", id: "g++", kind: "native", toolchain: ["g++"],
    sourceName: "hello.cpp",
    sample: `#include <iostream>\nint main(){std::cout<<"${HELLO}\\n";}\n`,
    steps: [["{tool:g++}", "{src}", "-o", "{out}"]],
  },
  {
    language: "C++", id: "clang++", kind: "native", toolchain: ["clang++"],
    sourceName: "hello.cpp",
    sample: `#include <iostream>\nint main(){std::cout<<"${HELLO}\\n";}\n`,
    steps: [["{tool:clang++}", "{src}", "-o", "{out}"]],
  },
  {
    language: "Objective-C", id: "gcc", kind: "native", toolchain: ["gcc"],
    sourceName: "hello.m",
    sample: `#include <stdio.h>\nint main(void){printf("${HELLO}\\n");return 0;}\n`,
    steps: [["{tool:gcc}", "-x", "objective-c", "{src}", "-o", "{out}"]],
    notes: "Foundation-free; GNUstep needed for full Obj-C runtime.",
  },
  {
    language: "Rust", id: "rustc", kind: "native", toolchain: ["rustc"],
    sourceName: "hello.rs",
    sample: `fn main(){println!("${HELLO}");}\n`,
    steps: [["{tool:rustc}", "{src}", "-o", "{out}"]],
  },
  {
    language: "Go", id: "go", kind: "native", toolchain: ["go"],
    sourceName: "hello.go",
    sample: `package main\nimport "fmt"\nfunc main(){fmt.Println("${HELLO}")}\n`,
    steps: [["{tool:go}", "build", "-o", "{out}", "{src}"]],
  },
  {
    language: "Zig", id: "zig", kind: "native", toolchain: ["zig"],
    sourceName: "hello.zig",
    sample: `const std=@import("std");\npub fn main() void { std.debug.print("${HELLO}\\n",.{}); }\n`,
    steps: [["{tool:zig}", "build-exe", "{src}", "-femit-bin={out}", "-target", "x86_64-linux"]],
  },
  {
    language: "Fortran", id: "gfortran", kind: "native", toolchain: ["gfortran"],
    sourceName: "hello.f90",
    sample: `program hello\n  print *, "${HELLO}"\nend program hello\n`,
    steps: [["{tool:gfortran}", "{src}", "-o", "{out}"]],
  },
  {
    language: "Ada", id: "gnatmake", kind: "native", toolchain: ["gnatmake"],
    sourceName: "main.adb",
    sample: `with Ada.Text_IO; procedure Main is begin Ada.Text_IO.Put_Line("${HELLO}"); end Main;\n`,
    steps: [["{tool:gnatmake}", "-o", "{out}", "{src}"]],
  },
  {
    language: "Pascal", id: "fpc", kind: "native", toolchain: ["fpc"],
    sourceName: "hello.pas",
    sample: `program Hello;\nbegin\n  writeln('${HELLO}');\nend.\n`,
    steps: [["{tool:fpc}", "-o{out}", "{src}"]],
  },
  {
    language: "D", id: "dmd", kind: "native", toolchain: ["dmd"],
    sourceName: "hello.d",
    sample: `import std.stdio;\nvoid main(){writeln("${HELLO}");}\n`,
    steps: [["{tool:dmd}", "-of{out}", "{src}"]],
  },
  {
    language: "D", id: "ldc2", kind: "native", toolchain: ["ldc2"],
    sourceName: "hello.d",
    sample: `import std.stdio;\nvoid main(){writeln("${HELLO}");}\n`,
    steps: [["{tool:ldc2}", "-of={out}", "{src}"]],
  },
  {
    language: "Nim", id: "nim", kind: "native", toolchain: ["nim"],
    sourceName: "hello.nim",
    sample: `echo "${HELLO}"\n`,
    steps: [["{tool:nim}", "c", "-d:release", "--nimcache:{dir}/nimcache", "-o:{out}", "{src}"]],
  },
  {
    language: "Crystal", id: "crystal", kind: "native", toolchain: ["crystal"],
    sourceName: "hello.cr",
    sample: `puts "${HELLO}"\n`,
    steps: [["{tool:crystal}", "build", "-o", "{out}", "{src}"]],
  },
  {
    language: "Swift", id: "swiftc", kind: "native", toolchain: ["swiftc"],
    sourceName: "hello.swift",
    sample: `print("${HELLO}")\n`,
    steps: [["{tool:swiftc}", "{src}", "-o", "{out}"]],
  },
  {
    language: "Haskell", id: "ghc", kind: "native", toolchain: ["ghc"],
    sourceName: "Main.hs",
    sample: `main :: IO ()\nmain = putStrLn "${HELLO}"\n`,
    steps: [["{tool:ghc}", "-O2", "-outputdir", "{dir}/hsobj", "-o", "{out}", "{src}"]],
  },
  {
    language: "OCaml", id: "ocamlopt", kind: "native", toolchain: ["ocamlopt"],
    sourceName: "hello.ml",
    sample: `let () = print_endline "${HELLO}"\n`,
    steps: [["{tool:ocamlopt}", "-o", "{out}", "{src}"]],
  },
  {
    language: "V", id: "v", kind: "native", toolchain: ["v"],
    sourceName: "hello.v",
    sample: `println('${HELLO}')\n`,
    steps: [["{tool:v}", "-o", "{out}", "{src}"]],
  },
  {
    language: "Odin", id: "odin", kind: "native", toolchain: ["odin"],
    sourceName: "hello.odin",
    sample: `package main\nimport "core:fmt"\nmain :: proc(){ fmt.println("${HELLO}") }\n`,
    steps: [["{tool:odin}", "build", "{src}", "-file", "-out:{out}"]],
  },
  {
    // GAS, x86-64, Linux syscalls — no libc, no headers.
    language: "Assembly", id: "gas", kind: "native", toolchain: ["as", "ld"],
    sourceName: "hello.s",
    sample:
      `.intel_syntax noprefix\n.section .data\nmsg: .ascii "${HELLO}\\n"\n.set len, . - msg\n` +
      `.section .text\n.globl _start\n_start:\n` +
      `  mov rax, 1\n  mov rdi, 1\n  lea rsi, [rip+msg]\n  mov rdx, len\n  syscall\n` +
      `  mov rax, 60\n  xor rdi, rdi\n  syscall\n`,
    steps: [
      ["{tool:as}", "-o", "{dir}/hello.o", "{src}"],
      ["{tool:ld}", "-o", "{out}", "{dir}/hello.o"],
    ],
  },
  {
    language: "Assembly", id: "nasm", kind: "native", toolchain: ["nasm", "ld"],
    sourceName: "hello.asm",
    sample:
      `section .data\nmsg db "${HELLO}",10\nlen equ $-msg\n` +
      `section .text\nglobal _start\n_start:\n` +
      `  mov rax,1\n  mov rdi,1\n  mov rsi,msg\n  mov rdx,len\n  syscall\n` +
      `  mov rax,60\n  xor rdi,rdi\n  syscall\n`,
    steps: [
      ["{tool:nasm}", "-f", "elf64", "-o", "{dir}/hello.o", "{src}"],
      ["{tool:ld}", "-o", "{out}", "{dir}/hello.o"],
    ],
  },
];

// Transpile / lower to C or LLVM-IR, then assemble to ELF.
const TRANSPILE: Strategy[] = [
  {
    language: "Vala", id: "valac", kind: "transpile", toolchain: ["valac"],
    sourceName: "hello.vala",
    sample: `void main(){ print("${HELLO}\\n"); }\n`,
    steps: [["{tool:valac}", "{src}", "-o", "{out}"]],
    notes: "valac lowers to C and invokes the C compiler.",
  },
  {
    language: "LLVM", id: "clang-ll", kind: "transpile", toolchain: ["clang"],
    sourceName: "hello.ll",
    sample:
      `@.s = private constant [13 x i8] c"${HELLO}\\0A\\00"\n` +
      `declare i32 @puts(i8*)\n` +
      `define i32 @main(){\n  %p = getelementptr [13 x i8], [13 x i8]* @.s, i64 0, i64 0\n` +
      `  call i32 @puts(i8* %p)\n  ret i32 0\n}\n`,
    steps: [["{tool:clang}", "{src}", "-o", "{out}"]],
    notes: "Hand-written textual LLVM IR compiled by clang.",
  },
];

// Ahead-of-time native compilation of otherwise-VM languages.
const AOT: Strategy[] = [
  {
    language: "Java", id: "native-image", kind: "aot", toolchain: ["javac", "native-image"],
    sourceName: "Hello.java",
    sample: `public class Hello{public static void main(String[] a){System.out.println("${HELLO}");}}\n`,
    steps: [
      ["{tool:javac}", "-d", "{dir}", "{src}"],
      ["{tool:native-image}", "-cp", "{dir}", "Hello", "{out}"],
    ],
    notes: "Needs GraalVM native-image on PATH.",
  },
  {
    language: "Kotlin", id: "kotlin-native", kind: "aot", toolchain: ["kotlinc-native"],
    sourceName: "hello.kt",
    sample: `fun main(){ println("${HELLO}") }\n`,
    steps: [["{tool:kotlinc-native}", "{src}", "-o", "{out}"]],
    artifact: "{out}.kexe",
    notes: "kotlinc-native (konanc) appends .kexe to the output name.",
  },
];

// Freeze an interpreter + program bytecode into a single ELF launcher.
const FREEZE: Strategy[] = [
  {
    language: "Python", id: "nuitka", kind: "freeze", toolchain: ["python3", "nuitka3"],
    sourceName: "hello.py",
    sample: `print("${HELLO}")\n`,
    steps: [["{tool:nuitka3}", "--onefile", "--output-dir={dir}/nuitka", "--output-filename={out}", "{src}"]],
  },
  {
    language: "Python", id: "pyinstaller", kind: "freeze", toolchain: ["pyinstaller"],
    sourceName: "hello.py",
    sample: `print("${HELLO}")\n`,
    steps: [["{tool:pyinstaller}", "--onefile", "--distpath", "{dir}/dist", "--workpath", "{dir}/build", "--specpath", "{dir}", "{src}"]],
    artifact: "{dir}/dist/hello",
  },
  {
    language: "JavaScript", id: "bun", kind: "freeze", toolchain: ["bun"],
    sourceName: "hello.js",
    sample: `console.log("${HELLO}");\n`,
    steps: [["{tool:bun}", "build", "{src}", "--compile", "--outfile", "{out}"]],
  },
  {
    language: "JavaScript", id: "deno", kind: "freeze", toolchain: ["deno"],
    sourceName: "hello.js",
    sample: `console.log("${HELLO}");\n`,
    steps: [["{tool:deno}", "compile", "--output", "{out}", "{src}"]],
  },
  {
    language: "TypeScript", id: "bun", kind: "freeze", toolchain: ["bun"],
    sourceName: "hello.ts",
    sample: `const m: string = "${HELLO}";\nconsole.log(m);\n`,
    steps: [["{tool:bun}", "build", "{src}", "--compile", "--outfile", "{out}"]],
  },
  {
    language: "TypeScript", id: "deno", kind: "freeze", toolchain: ["deno"],
    sourceName: "hello.ts",
    sample: `const m: string = "${HELLO}";\nconsole.log(m);\n`,
    steps: [["{tool:deno}", "compile", "--output", "{out}", "{src}"]],
  },
  {
    language: "Go", id: "tinygo", kind: "freeze", toolchain: ["tinygo"],
    sourceName: "hello.go",
    sample: `package main\nimport "fmt"\nfunc main(){fmt.Println("${HELLO}")}\n`,
    steps: [["{tool:tinygo}", "build", "-o", "{out}", "{src}"]],
  },
];

export const STRATEGIES: Strategy[] = [...NATIVE, ...TRANSPILE, ...AOT, ...FREEZE];

export function strategiesFor(language: string): Strategy[] {
  return STRATEGIES.filter(s => s.language.toLowerCase() === language.toLowerCase());
}

// ── ELF verification ───────────────────────────────────────────────────────
export interface ElfInfo {
  ok:      boolean;       // true iff a 64-bit little-endian x86-64 ELF
  class:   string;        // ELFCLASS32 | ELFCLASS64 | ?
  endian:  string;        // LE | BE | ?
  type:    string;        // ET_REL | ET_EXEC | ET_DYN | ET_CORE | ?
  machine: string;        // EM_X86_64 | EM_AARCH64 | … | 0x..
  reason?: string;
}

const E_TYPE: Record<number, string> = { 0: "ET_NONE", 1: "ET_REL", 2: "ET_EXEC", 3: "ET_DYN", 4: "ET_CORE" };
const E_MACHINE: Record<number, string> = {
  0x03: "EM_386", 0x28: "EM_ARM", 0x3e: "EM_X86_64", 0xb7: "EM_AARCH64", 0xf3: "EM_RISCV",
};

export function verifyElf(file: string): ElfInfo {
  const bad = (reason: string): ElfInfo =>
    ({ ok: false, class: "?", endian: "?", type: "?", machine: "?", reason });
  if (!fs.existsSync(file)) return bad("no output file");
  const fd = fs.openSync(file, "r");
  const buf = Buffer.alloc(20);
  const n = fs.readSync(fd, buf, 0, 20, 0);
  fs.closeSync(fd);
  if (n < 20) return bad("file too small");
  if (!(buf[0] === 0x7f && buf[1] === 0x45 && buf[2] === 0x4c && buf[3] === 0x46))
    return bad("no ELF magic");

  const cls    = buf[4] === 1 ? "ELFCLASS32" : buf[4] === 2 ? "ELFCLASS64" : "?";
  const endian = buf[5] === 1 ? "LE" : buf[5] === 2 ? "BE" : "?";
  const eType  = buf.readUInt16LE(16);
  const eMach  = buf.readUInt16LE(18);
  const type    = E_TYPE[eType] ?? `0x${eType.toString(16)}`;
  const machine = E_MACHINE[eMach] ?? `0x${eMach.toString(16)}`;

  const ok = buf[4] === 2 && buf[5] === 1 && eMach === 0x3e;
  return { ok, class: cls, endian, type, machine, reason: ok ? undefined : "not x86-64 / 64-bit / LE" };
}

// ── Build driver ─────────────────────────────────────────────────────────
export interface BuildResult {
  strategy: Strategy;
  ok:       boolean;
  artifact: string;
  elf?:     ElfInfo;
  missing:  string[];       // toolchain executables not found
  log:      string;         // combined stdout/stderr of the steps
  error?:   string;
}

function subst(template: string, vars: Record<string, string>): string {
  return template
    .replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? `{${k}}`)
    .replace(/\{tool:([^}]+)\}/g, (_, exe) => vars[`tool:${exe}`] ?? exe);
}

export interface BuildOpts {
  /** where to leave the ELF; default a fresh temp dir that is NOT cleaned up. */
  outDir?: string;
  keep?:   boolean;     // keep the work dir even on success (default true)
}

export function build(strategy: Strategy, opts: BuildOpts = {}): BuildResult {
  const missing = strategy.toolchain.filter(t => which(t) === null);
  const work = fs.mkdtempSync(path.join(opts.outDir ?? os.tmpdir(), `elf-${strategy.language}-${strategy.id}-`));
  const out = path.join(work, "out");
  const src = path.join(work, strategy.sourceName);

  const result: BuildResult = { strategy, ok: false, artifact: out, missing, log: "" };
  if (missing.length) {
    result.error = `missing toolchain: ${missing.join(", ")}`;
    return result;
  }

  const vars: Record<string, string> = { src, out, dir: work, base: strategy.sourceName.replace(/\.[^.]+$/, "") };
  for (const t of strategy.toolchain) vars[`tool:${t}`] = which(t)!;

  fs.writeFileSync(src, strategy.sample);

  let log = "";
  for (const step of strategy.steps) {
    const argv = step.map(a => subst(a, vars));
    const r = spawnSync(argv[0], argv.slice(1), { encoding: "utf8", cwd: work, maxBuffer: 32 * 1024 * 1024 });
    log += `$ ${argv.join(" ")}\n${r.stdout ?? ""}${r.stderr ?? ""}`;
    if (r.error)        { result.log = log; result.error = r.error.message; return result; }
    if (r.status !== 0) { result.log = log; result.error = `step exited ${r.status}`; return result; }
  }
  result.log = log;

  const artifact = subst(strategy.artifact ?? "{out}", vars);
  result.artifact = artifact;
  result.elf = verifyElf(artifact);
  result.ok  = result.elf.ok;
  if (!result.ok && !result.error) result.error = result.elf?.reason ?? "verification failed";
  return result;
}

// ── CLI ──────────────────────────────────────────────────────────────────
function installHint(language: string): string {
  const dir = languageDir();
  const sh = dir ? path.join(dir, language, "install.sh") : null;
  return sh && fs.existsSync(sh) ? `  install: bash "${sh}"` : "";
}

function cmdList(onlyAvailable: boolean) {
  const byLang = new Map<string, Strategy[]>();
  for (const s of STRATEGIES) (byLang.get(s.language) ?? byLang.set(s.language, []).get(s.language)!).push(s);
  for (const [lang, strs] of byLang) {
    const lines: string[] = [];
    for (const s of strs) {
      const missing = s.toolchain.filter(t => which(t) === null);
      if (onlyAvailable && missing.length) continue;
      const mark = missing.length ? `MISSING(${missing.join(",")})` : "ready";
      lines.push(`    ${s.id.padEnd(14)} ${s.kind.padEnd(10)} ${mark}`);
    }
    if (lines.length) { console.log(`${lang}`); console.log(lines.join("\n")); }
  }
}

function reportBuild(r: BuildResult) {
  const tag = r.ok ? "OK " : "FAIL";
  const where = r.ok ? `→ ${r.artifact}` : (r.error ?? "");
  const elf = r.elf && r.elf.machine !== "?" ? ` [${r.elf.class} ${r.elf.endian} ${r.elf.type} ${r.elf.machine}]` : "";
  console.log(`  ${tag} ${r.strategy.language}/${r.strategy.id}${elf} ${where}`);
  if (!r.ok && r.missing.length) {
    const hint = installHint(r.strategy.language);
    if (hint) console.log(hint);
  }
}

function cmdBuild(args: string[]) {
  let targets: Strategy[];
  if (args[0] === "--all" || args.length === 0) {
    targets = STRATEGIES;
  } else {
    const [lang, id] = args;
    targets = strategiesFor(lang);
    if (id) targets = targets.filter(s => s.id === id);
    if (!targets.length) { console.error(`No strategy for ${lang}${id ? "/" + id : ""}.`); process.exit(1); }
  }
  const results = targets.map(s => build(s));
  for (const r of results) reportBuild(r);
  const ok = results.filter(r => r.ok).length;
  const langOk = new Set(results.filter(r => r.ok).map(r => r.strategy.language)).size;
  console.log(`\n${ok}/${results.length} strategies produced an x86-64 ELF, covering ${langOk} languages.`);
  if (process.argv.includes("--log")) for (const r of results) if (!r.ok) console.error(`\n--- ${r.strategy.language}/${r.strategy.id} ---\n${r.log}`);
}

function cmdInstall(langs: string[]) {
  const dir = languageDir();
  if (!dir) { console.error("Cannot locate ray/@ether/library/Language."); process.exit(1); }
  const targets = langs.length ? langs : [...new Set(STRATEGIES.map(s => s.language))];
  for (const lang of targets) {
    const sh = path.join(dir, lang, "install.sh");
    if (!fs.existsSync(sh)) { console.log(`  SKIP ${lang} (no install.sh)`); continue; }
    console.log(`  RUN  ${lang}: ${sh}`);
    const r = spawnSync("bash", [sh], { stdio: "inherit" });
    console.log(r.status === 0 ? `  DONE ${lang}` : `  FAIL ${lang} (exit ${r.status})`);
  }
}

function cmdProbe() {
  const tools = [...new Set(STRATEGIES.flatMap(s => s.toolchain))].sort();
  for (const t of tools) {
    const p = which(t);
    console.log(`  ${t.padEnd(16)} ${p ?? "-"}`);
  }
}

if (process.argv[1] && import.meta.url === `file://${path.resolve(process.argv[1])}`) {
  const [cmd, ...rest] = process.argv.slice(2);
  switch (cmd) {
    case "list":  cmdList(rest.includes("--available")); break;
    case "build":   cmdBuild(rest.filter(a => a !== "--log")); break;
    case "install": cmdInstall(rest); break;
    case "probe":   cmdProbe(); break;
    default:
      console.log("usage: tsx src/elf.ts <list [--available] | build [<lang> [<id>] | --all] [--log] | install [<lang>...] | probe>");
  }
}
