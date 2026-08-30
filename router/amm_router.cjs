// Bachu et al. (2024) counterfactual, replicated: route each executed order
// through the Uniswap smart-order-router pinned to its settlement block against
// an archival RPC. CommonJS (the SOR is authored for CJS). Bridge for code.py.
//   node amm_router.cjs <input.json> <output.jsonl>
// input: [{id, tokenIn, tokenOut, amount, tradeType, block}, ...]
//   EXACT_INPUT  -> amount = tokenIn atoms,  quote = tokenOut out
//   EXACT_OUTPUT -> amount = tokenOut atoms, quote = tokenIn in
// output jsonl: {id, quote, gasUsed, route} or {id, error}. RPC via ETH_RPC_URL.
const fs = require('fs');
const { ethers } = require('ethers');
const { AlphaRouter, setGlobalLogger } = require('@uniswap/smart-order-router');
const { Token, CurrencyAmount, TradeType } = require('@uniswap/sdk-core');

const noop = () => {};
const silent = { trace: noop, debug: noop, info: noop, warn: noop, error: noop,
                 fatal: noop, child: () => silent };
setGlobalLogger(silent);

const RPC = process.env.ETH_RPC_URL;
if (!RPC) { console.error('ETH_RPC_URL not set'); process.exit(1); }
const CONCURRENCY = parseInt(process.env.ROUTER_CONCURRENCY || '2', 10);
// ethers v5 fetchJson auto-retries HTTP 429 with exponential backoff up to
// throttleLimit (honouring Retry-After); raise it well above the default (12)
// since the SOR fires many multicalls per route and Alchemy rate-limits.
const provider = new ethers.providers.JsonRpcProvider(
  { url: RPC, throttleLimit: 30, throttleSlotInterval: 250 }, 1);
const router = new AlphaRouter({ chainId: 1, provider });

const ERC20 = ['function decimals() view returns (uint8)'];
const decCache = new Map();
async function decimals(addr) {
  const k = addr.toLowerCase();
  if (decCache.has(k)) return decCache.get(k);
  let d = 18;
  try { d = Number(await new ethers.Contract(addr, ERC20, provider).decimals()); }
  catch (e) { d = 18; }
  decCache.set(k, d);
  return d;
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function routeOne(job) {
  let lastErr = '';
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      const din = await decimals(job.tokenIn);
      const dout = await decimals(job.tokenOut);
      const tin = new Token(1, ethers.utils.getAddress(job.tokenIn), din);
      const tout = new Token(1, ethers.utils.getAddress(job.tokenOut), dout);
      const exactIn = job.tradeType === 'EXACT_INPUT';
      const amount = exactIn ? CurrencyAmount.fromRawAmount(tin, job.amount)
                             : CurrencyAmount.fromRawAmount(tout, job.amount);
      const quoteCurrency = exactIn ? tout : tin;
      const tradeType = exactIn ? TradeType.EXACT_INPUT : TradeType.EXACT_OUTPUT;
      const route = await router.route(amount, quoteCurrency, tradeType, undefined,
                                       { blockNumber: job.block });
      if (!route) return { id: job.id, error: 'no_route' };
      return { id: job.id, quote: route.quote.quotient.toString(),
               gasUsed: route.estimatedGasUsed.toString(),
               route: route.route.map(r => r.protocol).join('+') };
    } catch (e) {
      lastErr = String(e && e.message ? e.message : e);
      // retry transient/rate-limit failures with backoff; give up on real no-route
      if (/quote|429|rate|timeout|network|econn/i.test(lastErr) && attempt < 3) {
        await sleep(500 * Math.pow(2, attempt) + Math.floor(Math.random() * 300));
        continue;
      }
      break;
    }
  }
  return { id: job.id, error: lastErr.slice(0, 120) };
}

async function main() {
  const [, , inPath, outPath] = process.argv;
  const jobs = JSON.parse(fs.readFileSync(inPath, 'utf8'));
  const out = fs.createWriteStream(outPath, { flags: 'w' });
  let done = 0, idx = 0;
  async function worker() {
    while (idx < jobs.length) {
      const r = await routeOne(jobs[idx++]);
      out.write(JSON.stringify(r) + '\n');
      if (++done % 10 === 0) console.error(`  routed ${done}/${jobs.length}`);
    }
  }
  await Promise.all(Array.from({ length: CONCURRENCY }, worker));
  await new Promise(res => out.end(res));
  console.error(`DONE ${done}/${jobs.length}`);
}
main().catch(e => { console.error(e); process.exit(1); });
