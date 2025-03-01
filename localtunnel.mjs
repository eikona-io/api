import ngrok from "@ngrok/ngrok";
import fs from "node:fs";
import path from "node:path";

// const port = process.env.PORT ? Number.parseInt(process.env.PORT) : 3000;
console.log(process.env.NGROK_AUTHTOKEN);
console.log(process.env.NGROK_DOMAIN);

let listener;
let url;

async function startTunnel() {
  try {
    listener = await ngrok.forward({
      addr: 3011,
      authtoken: process.env.NGROK_AUTHTOKEN,
      domain: process.env.NGROK_DOMAIN,
    });

    url = listener.url();
    const filePath = path.join(__dirname, "tunnel_url.txt");
    fs.writeFileSync(filePath, url, "utf8");

    console.log(`Ingress established at: ${listener.url()}`);
  } catch (error) {
    console.error('Failed to start tunnel:', error);
  }
}

async function checkTunnel() {
  try {
    if (!listener) {
      console.log('No tunnel exists, starting...');
      await startTunnel();
      return;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    try {
      const response = await fetch(url, {
        method: 'GET',
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      console.log(response.status, response.statusText);
      
      if (!response.ok) {
        console.log('Tunnel is down, restarting...');
        await startTunnel();
      }
    } catch (error) {
      clearTimeout(timeoutId);
      throw error; // Propagate the error to the outer catch block
    }
  } catch (error) {
    console.error('Error checking tunnel:', error);
    // Close existing listener if it exists
    if (listener) {
      try {
        await listener.close();
      } catch (closeError) {
        console.error('Error closing existing tunnel:', closeError);
      }
      listener = null;
    }
    await startTunnel();
  }
}

// Initial tunnel start
await startTunnel();

// Add sleep/wake detection
let lastCheckTime = Date.now();
const CHECK_INTERVAL = 5000;

setInterval(() => {
  const now = Date.now();
  const timeSinceLastCheck = now - lastCheckTime;
  
  // If the time since last check is significantly longer than our interval,
  // assume the system was sleeping
  if (timeSinceLastCheck > CHECK_INTERVAL * 2) {
    console.log('System resumed from sleep, checking tunnel...');
    checkTunnel();
  }
  
  lastCheckTime = now;
}, CHECK_INTERVAL);

// Keep the process alive
process.stdin.resume();
console.log("Press CTRL+C to exit");

process.on("SIGINT", () => {
  console.log("Exiting...");
  process.exit();
});