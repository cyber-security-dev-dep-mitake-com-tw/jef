/**
 * Fill PyPI's pending trusted-publisher form.
 *
 * Twelve identical forms -- six projects on pypi.org and six on test.pypi.org --
 * differing only in one field. This types the other four for you and picks the
 * right environment from the hostname, which is the field most likely to be got
 * wrong: the workflow routes prereleases to TestPyPI, so the two registries need
 * different environment names.
 *
 * It never submits. You read what it filled and click Add yourself.
 *
 * Usage: open the publishing page, paste this into DevTools console, then
 *
 *     jef.next()      fill the next project that has not been done
 *     jef.fill('jef-core')
 *     jef.status()
 *     jef.reset()
 *
 * Selectors come from warehouse's own template (form id
 * pending-github-publisher-form, fields by id), not from guesswork.
 */
(() => {
  const PROJECTS = ["jef", "jef-core", "jef-scene", "jef-server", "jef-sdk", "jef-train"];
  const OWNER = "cyber-security-dev-dep-mitake-com-tw";
  const REPOSITORY = "jef";
  // The filename, not the path. `.github/workflows/release.yml` is rejected in a
  // way that does not say which field is wrong.
  const WORKFLOW = "release.yml";

  const host = location.hostname;
  const isTest = host.startsWith("test.");
  const ENVIRONMENT = isTest ? "testpypi" : "pypi";
  const KEY = `jef-trusted-publisher:${host}`;

  const form = document.querySelector("#pending-github-publisher-form");
  if (!form) {
    console.error(
      "%cNo GitHub pending-publisher form on this page.",
      "color:#c00;font-weight:bold",
      "\nOpen https://pypi.org/manage/account/publishing/ (or the test.pypi.org one)" +
        " and make sure the GitHub tab is selected."
    );
    return;
  }

  const set = (id, value) => {
    const field = form.querySelector(`#${id}`);
    if (!field) throw new Error(`field #${id} is not on this form`);
    field.value = value;
    // Dispatched so any client-side validation sees the change rather than an
    // apparently untouched field.
    field.dispatchEvent(new Event("input", { bubbles: true }));
    field.dispatchEvent(new Event("change", { bubbles: true }));
    return value;
  };

  const done = () => JSON.parse(localStorage.getItem(KEY) || "[]");
  const markDone = (name) => {
    const current = done();
    if (!current.includes(name)) {
      localStorage.setItem(KEY, JSON.stringify([...current, name]));
    }
  };

  const fill = (name) => {
    if (!PROJECTS.includes(name)) {
      console.warn(`${name} is not one of: ${PROJECTS.join(", ")}`);
    }
    set("project_name", name);
    set("owner", OWNER);
    set("repository", REPOSITORY);
    set("workflow_filename", WORKFLOW);
    set("environment", ENVIRONMENT);
    markDone(name);

    console.log(
      `%cFilled ${name}%c on ${host}\n` +
        `  owner        ${OWNER}\n` +
        `  repository   ${REPOSITORY}\n` +
        `  workflow     ${WORKFLOW}\n` +
        `  environment  ${ENVIRONMENT}\n\n` +
        "Check it, then click Add. Re-paste this script after the reload and " +
        "run jef.next() for the next one.",
      "color:#080;font-weight:bold",
      "color:inherit"
    );
    return name;
  };

  const remaining = () => PROJECTS.filter((p) => !done().includes(p));

  const status = () => {
    const left = remaining();
    console.log(
      `%c${host}%c — ${PROJECTS.length - left.length}/${PROJECTS.length} filled\n` +
        PROJECTS.map((p) => `  ${done().includes(p) ? "✓" : "·"} ${p}`).join("\n") +
        (left.length
          ? `\n\nNext: jef.next()  →  ${left[0]}`
          : "\n\nAll six filled on this host. Do the other registry too."),
      "font-weight:bold",
      "color:inherit"
    );
    return left;
  };

  window.jef = {
    projects: PROJECTS,
    fill,
    next: () => {
      const left = remaining();
      if (!left.length) {
        console.log("All six already filled on this host.");
        return null;
      }
      return fill(left[0]);
    },
    status,
    // "Filled" is tracked locally and only means this script typed it. If a
    // submission failed, reset and redo that one.
    reset: () => {
      localStorage.removeItem(KEY);
      console.log(`Cleared progress for ${host}.`);
    },
  };

  console.log(
    `%cJEF trusted-publisher helper%c loaded for ${host} (environment: ${ENVIRONMENT})`,
    "color:#2d5f8a;font-weight:bold",
    "color:inherit"
  );
  status();
})();
