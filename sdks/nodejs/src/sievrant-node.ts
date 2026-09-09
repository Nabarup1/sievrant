import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const native = require('./sievrant-node.linux-x64-gnu.node');

export interface SievrantCorpus {
  create(name: string): Promise<SievrantCorpus>;
  open(name: string): Promise<SievrantCorpus>;
}

export const SievrantCorpus = native.SievrantCorpus as any;
