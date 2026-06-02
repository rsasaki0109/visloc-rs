/* Shared scene-picker bootstrap for splat.html / splat_spark.html / splat_webgpu.html.
 *
 * Each viewer supplies:
 *   - <select id="sceneSelect" data-testid="scene-picker">...</select>
 *   - optional <span id="sceneSummary"></span>
 *
 * This script:
 *   1. fetches ./scenes-list.json and, if the <select> is empty, populates it
 *      with <option value=url>label</option> entries (data-summary=summary).
 *   2. reflects the current ?url= back into the picker on load.
 *   3. on change, rewrites location.search so the viewer re-fetches.
 *   4. updates the sibling summary span, if one exists.
 *
 * docs/scenes-list.json is the source of truth. Some viewers pre-populate
 * <option>s for fast first paint; tests keep those options aligned with the
 * shared scene list.
 */
(function () {
  "use strict";
  var select = document.getElementById("sceneSelect");
  if (!select) return;
  var summary = document.getElementById("sceneSummary");

  function applySummary() {
    if (!summary) return;
    var opt = select.selectedOptions && select.selectedOptions[0];
    summary.textContent = opt ? (opt.dataset.summary || "") : "";
  }

  function applyCurrentUrl() {
    var params = new URLSearchParams(window.location.search);
    var current = params.get("url");
    if (!current) return;
    for (var i = 0; i < select.options.length; i++) {
      if (select.options[i].value === current) {
        select.value = current;
        return;
      }
    }
  }

  function attach() {
    applyCurrentUrl();
    applySummary();
    select.addEventListener("change", function () {
      var params = new URLSearchParams(window.location.search);
      params.set("url", select.value);
      window.location.assign(
        window.location.pathname + "?" + params.toString() + window.location.hash
      );
    });
  }

  if (select.options.length > 0) {
    attach();
    return;
  }

  // Empty select — try to populate from scenes-list.json.
  fetch("./scenes-list.json", { cache: "no-cache" })
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (data) {
      if (!data || !Array.isArray(data.scenes)) return;
      data.scenes.forEach(function (scene) {
        var opt = document.createElement("option");
        opt.value = scene.url;
        opt.textContent = scene.label || scene.url;
        if (scene.summary) opt.dataset.summary = scene.summary;
        select.appendChild(opt);
      });
    })
    .catch(function () { /* non-fatal; leave picker empty */ })
    .finally(attach);
})();
