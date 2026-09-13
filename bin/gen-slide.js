#!/usr/bin/env node

import { main } from "../lib/typstslideviewer.js";

main(process.argv.slice(2)).catch((error) => {
  console.error(`gen-slide: ${error.message}`);
  process.exitCode = 1;
});
