import { chromium } from '@playwright/test';

(async () => {
  console.log("Launching browser...");
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  // Set viewport to a nice standard size
  await page.setViewportSize({ width: 1280, height: 800 });

  console.log("Navigating to web demo...");
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(1000);

  console.log("Clicking sample contract...");
  await page.click('#sample-contract-btn');
  await page.waitForTimeout(500);

  console.log("Typing query...");
  await page.fill('#query-input', 'termination');
  await page.waitForTimeout(200);

  console.log("Clicking Query button...");
  await page.click('#ask-btn');
  
  console.log("Waiting for answer and citations...");
  await page.waitForSelector('.citation-chip', { timeout: 5000 });
  await page.waitForTimeout(500);

  console.log("Taking State 1 screenshot...");
  await page.screenshot({ path: 'scripts/state1.png' });

  console.log("Clicking citation chip...");
  await page.click('.citation-chip');
  await page.waitForTimeout(1000);

  console.log("Taking State 2 screenshot...");
  await page.screenshot({ path: 'scripts/state2.png' });

  console.log("Done!");
  await browser.close();
})();
