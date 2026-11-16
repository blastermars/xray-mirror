// Download + extract BodyParts3D 4.0 (DBCLS) into assets-src/bodyparts3d/ (cached; re-runs skip finished steps).
// Everything downloaded is treated as untrusted DATA: nothing in it is executed; only *.obj and *.txt are read.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
import { parseHeader } from './obj.mjs';

export const BASE_URL = 'https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/';
// isa archive = all 2234 ELEMENT meshes (the part_of archive holds a 1258-file subset with identical geometry).
export const ARCHIVE = 'isa_BP3D_4.0_obj_99.zip';
export const ARCHIVE_DIR = 'isa_BP3D_4.0_obj_99';
export const LISTS = ['isa_element_parts.txt', 'partof_element_parts.txt', 'isa_parts_list_e.txt', 'partof_parts_list_e.txt'];

function download(url, dest) {
  const tmp = dest + '.part';
  console.log(`  downloading ${url}`);
  execFileSync('curl', ['-L', '--fail', '--silent', '--show-error', '--retry', '3', '-o', tmp, url], { stdio: 'inherit' });
  fs.renameSync(tmp, dest);
}

export function ensureSource(srcDir) {
  fs.mkdirSync(srcDir, { recursive: true });
  for (const f of [...LISTS, ARCHIVE]) {
    const p = path.join(srcDir, f);
    if (!fs.existsSync(p) || fs.statSync(p).size === 0) download(BASE_URL + f, p);
  }
  const objDir = path.join(srcDir, ARCHIVE_DIR);
  const marker = path.join(objDir, '.extracted');
  if (!fs.existsSync(marker)) {
    console.log(`  extracting ${ARCHIVE}`);
    // only the .obj members; system unzip refuses absolute paths / strips ../ components
    execFileSync('unzip', ['-q', '-o', path.join(srcDir, ARCHIVE), `${ARCHIVE_DIR}/*.obj`, '-d', srcDir], { stdio: 'inherit' });
    fs.writeFileSync(marker, new Date().toISOString());
  }
  return objDir;
}

/** Index of every element file: header metadata (id, FMA concept, English name, bounds). Cached as JSON. */
export function elementIndex(srcDir, objDir) {
  const cache = path.join(srcDir, 'element_index.json');
  if (fs.existsSync(cache)) return JSON.parse(fs.readFileSync(cache, 'utf8'));
  const out = [];
  for (const f of fs.readdirSync(objDir).filter((x) => /^FJ\d+M?\.obj$/.test(x)).sort()) {
    const fd = fs.openSync(path.join(objDir, f), 'r');
    const buf = Buffer.alloc(4096); const n = fs.readSync(fd, buf, 0, 4096, 0); fs.closeSync(fd);
    const h = parseHeader(buf.toString('utf8', 0, n));
    out.push({ file: f, ...h });
  }
  fs.writeFileSync(cache, JSON.stringify(out));
  return out;
}

/** Map FJ id -> Set of FMA concept ids from the element_parts lists (both trees), for cross-checking headers. */
export function elementParts(srcDir) {
  const map = new Map();
  for (const f of ['isa_element_parts.txt', 'partof_element_parts.txt']) {
    const lines = fs.readFileSync(path.join(srcDir, f), 'utf8').split(/\r?\n/).slice(1);
    for (const l of lines) {
      const [fma, , fj] = l.split('\t');
      if (!fj) continue;
      if (!map.has(fj)) map.set(fj, new Set());
      map.get(fj).add(fma);
    }
  }
  return map;
}

// ---- BodyParts3D 3.0 (2011-09-15 release): source of facial-expression / masticatory muscles.
export const BASE_URL_30 = 'https://dbarchive.biosciencedbc.jp/data/bodyparts3d/20110915/';
export const ARCHIVE_30 = 'BodyParts3D_3.0_obj_99.zip';
export const LISTS_30 = ['parts_list_e.txt', 'composite_parts.txt', 'conventional_part_of.txt', 'README_e.html'];

export function ensureSource30(srcDir) {
  fs.mkdirSync(srcDir, { recursive: true });
  for (const f of [...LISTS_30, ARCHIVE_30]) {
    const p = path.join(srcDir, f);
    if (!fs.existsSync(p) || fs.statSync(p).size === 0) download(BASE_URL_30 + f, p);
  }
  const objDir = path.join(srcDir, 'obj');
  const marker = path.join(objDir, '.extracted');
  if (!fs.existsSync(marker)) {
    console.log(`  extracting ${ARCHIVE_30}`);
    fs.mkdirSync(objDir, { recursive: true });
    // -j: flatten member paths; only *.obj members. The archive stores Windows "\" separators, which makes
    // unzip exit with status 1 (warning) after extracting correctly; anything else is an error.
    const r = spawnSync('unzip', ['-q', '-o', '-j', path.join(srcDir, ARCHIVE_30), '*.obj', '-d', objDir], { stdio: 'inherit' });
    if (r.status !== 0 && r.status !== 1) throw new Error(`unzip failed with status ${r.status}`);
    fs.writeFileSync(marker, new Date().toISOString());
  }
  return objDir;
}

/** 3.0 element index: concept id (file name) -> {file, id, name} using parts_list_e.txt for names. */
export function elementIndex30(srcDir, objDir) {
  const names = new Map();
  for (const l of fs.readFileSync(path.join(srcDir, 'parts_list_e.txt'), 'utf8').split(/\r?\n/).slice(1)) {
    const [id, en] = l.split('\t'); if (id && en) names.set(id.replace(/"/g, ''), en.trim());
  }
  const out = [];
  for (const f of fs.readdirSync(objDir).filter((x) => x.endsWith('.obj')).sort()) {
    const id = f.replace(/\.obj$/, '');
    const fd = fs.openSync(path.join(objDir, f), 'r');
    const buf = Buffer.alloc(4096); const n = fs.readSync(fd, buf, 0, 4096, 0); fs.closeSync(fd);
    const h = parseHeader(buf.toString('utf8', 0, n));
    out.push({ file: f, id, fma: id, name: names.get(id) || h.name || '', headerName: h.name });
  }
  return out;
}
