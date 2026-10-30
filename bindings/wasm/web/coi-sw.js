// Static hosts such as GitHub Pages cannot send the COOP/COEP headers that
// make the page cross-origin isolated, which wasm threads need
// (SharedArrayBuffer). This service worker adds them to the page and its
// workers. index.html registers it only where the page is not isolated
// already (serve.py sends the headers itself). Other requests pass untouched:
// same-origin subresources need no headers.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.mode !== "navigate" && request.destination !== "worker") return;
  event.respondWith(
    fetch(request).then((response) => {
      if (response.status === 0) return response;
      const headers = new Headers(response.headers);
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      headers.set("Cross-Origin-Embedder-Policy", "require-corp");
      return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
    }),
  );
});
